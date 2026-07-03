"""
Power Balance Residual
=======================
Checks averaged power conservation:
  Residual_PWR = Vin * IL * D - Vout^2 / Rload
"""

import torch


def compute_power_balance_residual(
    IL: torch.Tensor,
    Vout: torch.Tensor,
    Vin: torch.Tensor,
    D: torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """Calculate normalized power balance violation."""
    P_in = Vin * IL
    P_out = (Vout ** 2) / (Rload + 1e-12)
    residual = P_in - P_out
    return torch.mean(residual ** 2)
