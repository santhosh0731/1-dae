"""
PINN Loss Functions
====================
All 6 loss components for the Physics-Informed Neural Network:

  L_total = λ_data * L_data
          + λ_kvl  * L_KVL
          + λ_kcl  * L_KCL
          + λ_dae  * L_DAE
          + λ_bc   * L_BC
          + λ_ic   * L_IC
          + λ_pwr  * L_pwr
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from phase4_pinn.physics.residuals import all_residuals


class PINNLoss(nn.Module):
    """
    Combined PINN loss with data + physics + boundary + initial conditions.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)

    def data_loss(
        self,
        pred:   torch.Tensor,  # (B, 5) predicted [Vout, IL, Vc, dIL, dVc]
        target: torch.Tensor,  # (B, 3) ground truth [Vout, IL, Vc]
    ) -> torch.Tensor:
        """MSE between predicted and LTspice ground truth."""
        return self.mse(pred[:, :3], target)

    def physics_losses(
        self,
        pred:  torch.Tensor,   # (B, 5)
        params: torch.Tensor,  # (B, 4) [Vin, D, L, C, Rload]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KVL, KCL, DAE, and power balance residuals.
        Uses explicitly predicted derivatives (cols 3,4) instead of autograd.
        """
        Vout   = pred[:, 0]
        IL     = pred[:, 1]
        Vc     = pred[:, 2]
        dIL_dt = pred[:, 3]
        dVc_dt = pred[:, 4]

        Vin   = params[:, 0]
        D     = params[:, 1]
        L     = params[:, 2]
        C     = params[:, 3]
        Rload = params[:, 4]

        res = all_residuals(IL, Vc, Vout, dIL_dt, dVc_dt, Vin, D, L, C, Rload)

        return {
            'kvl': res['kvl_norm'],
            'kcl': res['kcl_norm'],
            'dae': res['dae_norm'],
            'pwr': res['pwr_norm'],
        }

    def boundary_loss(
        self,
        pred_final:   torch.Tensor,  # (N_steps, 5) predictions at t=T
        target_final: torch.Tensor,  # (N_steps, 3) steady-state targets
    ) -> torch.Tensor:
        """
        Boundary condition loss: ensure predictions reach steady-state.
          pred[Vout, IL, Vc] at t=T ≈ [Vout_ss, IL_ss, Vc_ss]
        """
        if pred_final.shape[0] == 0:
            return torch.tensor(0.0)
        return self.mse(pred_final[:, :3], target_final)

    def initial_condition_loss(
        self,
        pred_init:   torch.Tensor,   # (N_steps, 5) predictions at t=0
        target_init: torch.Tensor,   # (N_steps, 3) [IL0, Vout0, Vc0]
    ) -> torch.Tensor:
        """
        Initial condition loss: predictions at t=0 match converter initial state.
        """
        if pred_init.shape[0] == 0:
            return torch.tensor(0.0)
        return self.mse(pred_init[:, :3], target_init)

    def total_loss(
        self,
        pred:         torch.Tensor,
        target:       torch.Tensor,
        params:       torch.Tensor,
        pred_bc:      Optional[torch.Tensor] = None,
        target_bc:    Optional[torch.Tensor] = None,
        pred_ic:      Optional[torch.Tensor] = None,
        target_ic:    Optional[torch.Tensor] = None,
        weights:      Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total weighted PINN loss.

        Returns:
            total_loss: scalar tensor
            loss_dict:  dict of individual loss components
        """
        if weights is None:
            weights = {'data': 1.0, 'kvl': 1.0, 'kcl': 1.0,
                       'dae': 0.5,  'bc': 0.5,  'ic': 0.5, 'pwr': 0.3}

        loss_dict = {}

        # Data loss
        loss_dict['data'] = self.data_loss(pred, target)

        # Physics losses
        phys = self.physics_losses(pred, params)
        loss_dict.update(phys)

        # Boundary condition loss
        if pred_bc is not None and target_bc is not None:
            loss_dict['bc'] = self.boundary_loss(pred_bc, target_bc)
        else:
            loss_dict['bc'] = torch.tensor(0.0, device=pred.device)

        # Initial condition loss
        if pred_ic is not None and target_ic is not None:
            loss_dict['ic'] = self.initial_condition_loss(pred_ic, target_ic)
        else:
            loss_dict['ic'] = torch.tensor(0.0, device=pred.device)

        # Weighted total
        total = torch.tensor(0.0, device=pred.device)
        for key, loss_val in loss_dict.items():
            w = weights.get(key, 0.0)
            total = total + w * loss_val

        loss_dict['total'] = total
        return total, loss_dict
