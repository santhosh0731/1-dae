"""
Physics Residuals — Aggregated Module
=======================================
Combines KVL, KCL, DAE, and power balance residuals
into a single callable for the PINN loss function.
"""

import torch
from typing import Dict


# ── KVL / Inductor ────────────────────────────────────────────────────────────

def kvl_residual(
    dIL_dt: torch.Tensor,
    Vin:    torch.Tensor,
    D:      torch.Tensor,
    Vout:   torch.Tensor,
    L:      torch.Tensor,
) -> torch.Tensor:
    """
    KVL / Inductor ODE residual:
      r_KVL = L * dIL/dt - (Vin - (1-D)*Vout)

    Should be zero everywhere along correct trajectories.
    """
    return L * dIL_dt - (Vin - (1.0 - D) * Vout)


# ── KCL / Capacitor ───────────────────────────────────────────────────────────

def kcl_residual(
    dVc_dt: torch.Tensor,
    IL:     torch.Tensor,
    D:      torch.Tensor,
    Vc:     torch.Tensor,
    C:      torch.Tensor,
    Rload:  torch.Tensor,
) -> torch.Tensor:
    """
    KCL / Capacitor ODE residual:
      r_KCL = C * dVc/dt - ((1-D)*IL - Vc/Rload)
    """
    return C * dVc_dt - ((1.0 - D) * IL - Vc / Rload)


# ── Algebraic Constraint ──────────────────────────────────────────────────────

def algebraic_residual(
    Vout: torch.Tensor,
    Vc:   torch.Tensor,
) -> torch.Tensor:
    """
    DAE algebraic constraint:
      g1 = Vout - Vc = 0
    """
    return Vout - Vc


# ── Power Balance ─────────────────────────────────────────────────────────────

def power_balance_residual(
    Vin:   torch.Tensor,
    IL:    torch.Tensor,
    D:     torch.Tensor,
    Vout:  torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """
    Power balance residual:
      r_pwr = Vin*IL*D - Vout^2/Rload

    At ideal steady-state Pin = Pout → residual = 0.
    """
    P_in  = Vin * IL * D
    P_out = Vout ** 2 / Rload
    return P_in - P_out


# ── Physics Feature Engineering ───────────────────────────────────────────────

def compute_physics_features(
    IL:    torch.Tensor,
    Vout:  torch.Tensor,
    Vin:   torch.Tensor,
    D:     torch.Tensor,
    L:     torch.Tensor,
    C:     torch.Tensor,
    Rload: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute additional physics features for feature-engineered dataset.

    Returns:
      E_L           : Inductor stored energy  0.5*L*IL^2   [J]
      E_C           : Capacitor stored energy 0.5*C*Vout^2 [J]
      P_in          : Input power  Vin*IL*D                 [W]
      P_out         : Output power Vout^2/Rload             [W]
      E_conserv_err : |P_in - P_out| / P_in                [%]
      efficiency    : P_out / P_in * 100                   [%]
      duty_region   : 0=light (D<0.5), 1=heavy (D>=0.5)
    """
    E_L           = 0.5 * L * IL ** 2
    E_C           = 0.5 * C * Vout ** 2
    P_in          = Vin * IL * D + 1e-9                    # avoid /0
    P_out         = Vout ** 2 / (Rload + 1e-9)
    E_conserv_err = torch.abs(P_in - P_out) / P_in * 100.0
    efficiency    = torch.clamp(P_out / P_in * 100.0, 0.0, 100.0)
    duty_region   = (D >= 0.5).float()

    return {
        'E_L':           E_L,
        'E_C':           E_C,
        'P_in':          P_in,
        'P_out':         P_out,
        'E_conserv_err': E_conserv_err,
        'efficiency':    efficiency,
        'duty_region':   duty_region,
    }


# ── Aggregate ─────────────────────────────────────────────────────────────────

def all_residuals(
    IL:     torch.Tensor,
    Vc:     torch.Tensor,
    Vout:   torch.Tensor,
    dIL_dt: torch.Tensor,
    dVc_dt: torch.Tensor,
    Vin:    torch.Tensor,
    D:      torch.Tensor,
    L:      torch.Tensor,
    C:      torch.Tensor,
    Rload:  torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute all physics residuals in one call.
    Used by loss_functions.py.
    """
    r_kvl = kvl_residual(dIL_dt, Vin, D, Vout, L)
    r_kcl = kcl_residual(dVc_dt, IL, D, Vc, C, Rload)
    r_dae = algebraic_residual(Vout, Vc)
    r_pwr = power_balance_residual(Vin, IL, D, Vout, Rload)

    return {
        'r_kvl':     r_kvl,
        'r_kcl':     r_kcl,
        'r_dae':     r_dae,
        'r_pwr':     r_pwr,
        'kvl_norm':  r_kvl.pow(2).mean(),
        'kcl_norm':  r_kcl.pow(2).mean(),
        'dae_norm':  r_dae.pow(2).mean(),
        'pwr_norm':  r_pwr.pow(2).mean(),
    }
