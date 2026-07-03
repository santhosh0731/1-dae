"""
Kirchhoff's Voltage Law (KVL) Residual
========================================
Checks dynamic inductor voltage matching:
  Residual_KVL = L * dIL_dt - (Vin - (1 - D) * Vout)
"""

import torch


def compute_kvl_residual(
    dIL_dt: torch.Tensor,
    Vout: torch.Tensor,
    Vin: torch.Tensor,
    D: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """Calculate normalized KVL violation."""
    v_ind = L * dIL_dt
    v_applied = Vin - (1.0 - D) * Vout
    residual = v_ind - v_applied
    return torch.mean(residual ** 2)
