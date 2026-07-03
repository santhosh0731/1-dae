"""
Unified Physics Residuals Manager
==================================
Combines KVL, KCL, DAE, power balance, and energy conservation residuals.
"""

import torch
from typing import Dict

from phase5_dae_pinn.physics.kvl import compute_kvl_residual
from phase5_dae_pinn.physics.kcl import compute_kcl_residual_fixed as compute_kcl_residual
from phase5_dae_pinn.physics.power_balance import compute_power_balance_residual
from phase5_dae_pinn.physics.energy_conservation import compute_energy_conservation_residual


from typing import Dict, Optional


def compute_all_residuals(
    pred: torch.Tensor,     # (B, 5) [Vout, IL, Vc, dIL_dt, dVc_dt]
    params: torch.Tensor,   # (B, 5) [Vin, D, L, C, Rload]
    scales: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all physics residuals, normalized by dataset reference scales.
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

    # Reference scales
    V_REF = scales['V_REF'] if scales is not None else 114.5
    I_REF = scales['I_REF'] if scales is not None else 380.0
    P_REF = scales['P_REF'] if scales is not None else 43510.0

    loss_kvl = compute_kvl_residual(dIL_dt, Vout, Vin, D, L) / (V_REF ** 2)
    loss_kcl = compute_kcl_residual(dVc_dt, IL, Vc, D, C, Rload) / (I_REF ** 2)
    loss_dae = torch.mean(((Vout - Vc) / V_REF) ** 2)
    loss_pwr = compute_power_balance_residual(IL, Vout, Vin, D, Rload) / (P_REF ** 2)
    loss_nrg = compute_energy_conservation_residual(IL, Vc, dIL_dt, dVc_dt, Vin, D, L, C, Rload) / (P_REF ** 2)

    return {
        'kvl': loss_kvl,
        'kcl': loss_kcl,
        'dae': loss_dae,
        'pwr': loss_pwr,
        'nrg': loss_nrg,
    }
