"""
Energy Conservation Residual
=============================
Checks dynamic energy rate vs power difference:
  dE_dt = d/dt (0.5 * L * IL^2 + 0.5 * C * Vc^2)
        = L * IL * dIL_dt + C * Vc * dVc_dt
  P_diff = Vin * IL * D - Vc^2 / Rload
  Residual = dE_dt - P_diff
"""

import torch


def compute_energy_conservation_residual(
    IL: torch.Tensor,
    Vc: torch.Tensor,
    dIL_dt: torch.Tensor,
    dVc_dt: torch.Tensor,
    Vin: torch.Tensor,
    D: torch.Tensor,
    L: torch.Tensor,
    C: torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """Calculate energy conservation law violation."""
    dE_dt = L * IL * dIL_dt + C * Vc * dVc_dt
    P_in = Vin * IL
    P_out = (Vc ** 2) / (Rload + 1e-12)
    P_diff = P_in - P_out
    residual = dE_dt - P_diff
    return torch.mean(residual ** 2)
