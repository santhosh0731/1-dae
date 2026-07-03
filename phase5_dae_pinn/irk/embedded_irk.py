"""
Embedded Implicit Runge-Kutta (Radau-IIA) Layer
================================================
Differentiable PyTorch Collocation Layer implementing the 3-stage Radau-IIA
implicit integration system (order 5) for stiff DAE training.

Butcher Tableau for 3-stage Radau-IIA:
  c = [ (4-sqrt(6))/10, (4+sqrt(6))/10, 1.0 ]
  a = [
    [ (88-7*sqrt(6))/360,   (296-169*sqrt(6))/1800, (-2+3*sqrt(6))/225 ],
    [ (296+169*sqrt(6))/1800, (88+7*sqrt(6))/360,     (-2-3*sqrt(6))/225 ],
    [ (16-sqrt(6))/36,        (16+sqrt(6))/36,        1/9                ]
  ]
  b = [ (16-sqrt(6))/36, (16+sqrt(6))/36, 1/9 ] (stiffly accurate: b_i = a_3i)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

# Radau-IIA Coefficients (double precision constants)
SQ6 = np.sqrt(6.0)
C_RADAU = [ (4.0 - SQ6)/10.0, (4.0 + SQ6)/10.0, 1.0 ]

A_RADAU = [
    [ (88.0 - 7.0*SQ6)/360.0,     (296.0 - 169.0*SQ6)/1800.0,  (-2.0 + 3.0*SQ6)/225.0 ],
    [ (296.0 + 169.0*SQ6)/1800.0, (88.0 + 7.0*SQ6)/360.0,      (-2.0 - 3.0*SQ6)/225.0 ],
    [ (16.0 - SQ6)/36.0,          (16.0 + SQ6)/36.0,           1.0/9.0                ]
]

B_RADAU = [ (16.0 - SQ6)/36.0, (16.0 + SQ6)/36.0, 1.0/9.0 ]


class DifferentiableRadauIIALayer(nn.Module):
    """
    Differentiable Radau-IIA Collocation residual layer.
    """

    def __init__(self, model: nn.Module, step_size_h: float = 1e-5):
        super().__init__()
        self.model = model
        self.h = step_size_h

        # Register tableau coefficients as PyTorch constants
        self.register_buffer('c', torch.tensor(C_RADAU, dtype=torch.float32))
        self.register_buffer('a', torch.tensor(A_RADAU, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(B_RADAU, dtype=torch.float32))

    def evaluate_f(
        self,
        X_state: torch.Tensor,   # (B, 2) [IL, Vc]
        Z_state: torch.Tensor,   # (B, 1) [Vout]
        Vin: torch.Tensor,
        D: torch.Tensor,
        L: torch.Tensor,
        C: torch.Tensor,
        Rload: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate state derivative f(x,z,u) using raw parameters."""
        IL, Vc = X_state[:, 0], X_state[:, 1]
        Vout = Z_state[:, 0]

        dIL_dt = (Vin - (1.0 - D) * Vout) / (L + 1e-12)
        dVc_dt = ((1.0 - D) * IL - Vc / (Rload + 1e-12)) / (C + 1e-12)

        return torch.stack([dIL_dt, dVc_dt], dim=1)  # (B, 2)

    def forward(
        self,
        inputs: torch.Tensor,  # (B, 7) [t, Vin, D, Fs, L, C, Rload] (normalized)
        pred_t0: torch.Tensor, # (B, 5) predicted outputs at t (raw physical values)
        params: torch.Tensor,  # (B, 5) [Vin, D, L, C, Rload] (raw physical values)
        t_std: float,          # Standard deviation of time for scaling h
    ) -> torch.Tensor:
        """
        Compute Radau-IIA collocation residual using physical parameters and scaled time steps.
        """
        # States at t_n
        x_n = pred_t0[:, [1, 2]]  # [IL, Vc]

        # Extract raw parameter inputs
        Vin   = params[:, 0]
        D     = params[:, 1]
        L     = params[:, 2]
        C     = params[:, 3]
        Rload = params[:, 4]

        # Standard parameters for feeding back to inputs
        Fs = inputs[:, 3]
        Vin_norm = inputs[:, 1]
        D_norm = inputs[:, 2]
        L_norm = inputs[:, 4]
        C_norm = inputs[:, 5]
        Rload_norm = inputs[:, 6]

        # Scaled step size for normalized time axis
        h_norm = self.h / t_std

        # Compute predictions at intermediate stages: t_i = t_n + c_i * h
        f_stages = []
        for i in range(3):
            t_stage = inputs[:, 0] + self.c[i] * h_norm

            # Construct input vector for stage i (time is scaled, others are normalized parameters)
            inputs_stage = torch.stack([t_stage, Vin_norm, D_norm, Fs, L_norm, C_norm, Rload_norm], dim=1)

            # Predict states at stage i
            pred_stage = self.model(inputs_stage)
            Y_stage = pred_stage[:, [1, 2]]  # [IL_stage, Vc_stage]
            Z_stage = pred_stage[:, [0]]     # [Vout_stage]

            # Evaluate state derivative f(Y_stage, Z_stage, u)
            f_val = self.evaluate_f(Y_stage, Z_stage, Vin, D, L, C, Rload)
            f_stages.append(f_val)

        # Predict state at next step: t_{n+1} = t_n + h
        t_next = inputs[:, 0] + h_norm
        inputs_next = torch.stack([t_next, Vin_norm, D_norm, Fs, L_norm, C_norm, Rload_norm], dim=1)
        pred_next = self.model(inputs_next)
        x_next_pred = pred_next[:, [1, 2]]  # [IL_next, Vc_next]

        # Compute Radau-IIA integrated state:
        # x_integrated = x_n + h * sum(b_i * f_i)
        sum_b_f = torch.zeros_like(x_n)
        for i in range(3):
            sum_b_f = sum_b_f + self.b[i] * f_stages[i]

        x_integrated = x_n + self.h * sum_b_f

        # Implicit RK integration residual error
        irk_residual = x_next_pred - x_integrated
        return irk_residual
