"""
Differential-Algebraic Equation (DAE) Formulation
==================================================
Formulates the boost converter averaged model as a semi-explicit index-1 DAE system:
  dx_dt = f(x, z, u, params)
  0     = g(x, z, u, params)

Differential states x:    [IL, Vc]
Algebraic variables z:    [Vout]
Control variables u:      [Vin, D]
Physics parameters:       [L, C, Rload]
"""

import numpy as np
import torch
from typing import Tuple, Dict, Union


def evaluate_dae_numpy(
    x: np.ndarray,      # (N, 2) [IL, Vc]
    z: np.ndarray,      # (N, 1) [Vout]
    dx_dt: np.ndarray,  # (N, 2) [dIL_dt, dVc_dt]
    u: np.ndarray,      # (N, 2) [Vin, D]
    params: np.ndarray, # (N, 3) [L, C, Rload]
) -> Dict[str, np.ndarray]:
    """Evaluate DAE system residuals using NumPy (for solvers & metrics)."""
    IL, Vc = x[:, 0], x[:, 1]
    Vout = z[:, 0]
    dIL_dt, dVc_dt = dx_dt[:, 0], dx_dt[:, 1]

    Vin, D = u[:, 0], u[:, 1]
    L, C, Rload = params[:, 0], params[:, 1], params[:, 2]

    # dx_dt - f(x, z, u)
    res_IL = dIL_dt - (Vin - (1.0 - D) * Vout) / (L + 1e-12)
    res_Vc = dVc_dt - ((1.0 - D) * IL - Vc / (Rload + 1e-12)) / (C + 1e-12)

    # g(x, z, u)
    res_alg = Vout - Vc

    return {
        'diff': np.column_stack([res_IL, res_Vc]),
        'alg': res_alg.reshape(-1, 1),
    }


def evaluate_dae_torch(
    x: torch.Tensor,      # (B, 2) [IL, Vc]
    z: torch.Tensor,      # (B, 1) [Vout]
    dx_dt: torch.Tensor,  # (B, 2) [dIL_dt, dVc_dt]
    u: torch.Tensor,      # (B, 2) [Vin, D]
    params: torch.Tensor, # (B, 3) [L, C, Rload]
) -> Dict[str, torch.Tensor]:
    """Evaluate DAE system residuals using PyTorch (for loss backprop)."""
    IL = x[:, 0]
    Vc = x[:, 1]
    Vout = z[:, 0]
    dIL_dt = dx_dt[:, 0]
    dVc_dt = dx_dt[:, 1]

    Vin = u[:, 0]
    D = u[:, 1]
    L = params[:, 0]
    C = params[:, 1]
    Rload = params[:, 2]

    res_IL = dIL_dt - (Vin - (1.0 - D) * Vout) / (L + 1e-12)
    res_Vc = dVc_dt - ((1.0 - D) * IL - Vc / (Rload + 1e-12)) / (C + 1e-12)
    res_alg = Vout - Vc

    return {
        'diff': torch.stack([res_IL, res_Vc], dim=1),
        'alg': res_alg.unsqueeze(1),
    }
