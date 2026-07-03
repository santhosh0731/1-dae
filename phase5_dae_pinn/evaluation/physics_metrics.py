"""
Physics Consistency Metrics
============================
Calculates physical violations: KVL, KCL, DAE constraints, power conservation.
"""

import numpy as np
from typing import Dict


def compute_physics_violations(
    pred_real: np.ndarray,   # (N, 5) [Vout, IL, Vc, dIL_dt, dVc_dt]
    params: np.ndarray,      # (N, 5) [Vin, D, L, C, Rload]
) -> Dict[str, float]:
    Vout   = pred_real[:, 0]
    IL     = pred_real[:, 1]
    Vc     = pred_real[:, 2]
    dIL_dt = pred_real[:, 3]
    dVc_dt = pred_real[:, 4]

    Vin   = params[:, 0]
    D     = params[:, 1]
    L     = params[:, 2]
    C     = params[:, 3]
    Rload = params[:, 4]

    # KVL: L * dIL/dt - (Vin - (1-D)*Vout) = 0
    res_kvl = L * dIL_dt - (Vin - (1.0 - D) * Vout)

    # KCL: C * dVc/dt - ((1-D)*IL - Vc/Rload) = 0
    res_kcl = C * dVc_dt - ((1.0 - D) * IL - Vc / (Rload + 1e-12))

    # DAE: Vout - Vc = 0
    res_dae = Vout - Vc

    # Power balance
    P_in = Vin * IL
    P_out = (Vout ** 2) / (Rload + 1e-12)
    res_pwr = P_in - P_out

    # Energy conservation
    dE_dt = L * IL * dIL_dt + C * Vc * dVc_dt
    res_nrg = dE_dt - (P_in - P_out)

    return {
        'KVL_residual_norm':    float(np.sqrt(np.mean(res_kvl**2))),
        'KCL_residual_norm':    float(np.sqrt(np.mean(res_kcl**2))),
        'DAE_constraint_error': float(np.sqrt(np.mean(res_dae**2))),
        'Power_balance_RMSE':   float(np.sqrt(np.mean(res_pwr**2))),
        'Energy_conserv_err':   float(np.sqrt(np.mean(res_nrg**2))),
    }
