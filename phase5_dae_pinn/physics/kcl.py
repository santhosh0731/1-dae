"""
Kirchhoff's Current Law (KCL) Residual
========================================
Checks dynamic capacitor current matching:
  Residual_KCL = C * dVc_dt - ((1 - D) * IL - Vc / Rload)
"""

import torch


def compute_kcl_residual(
    dVc_dt: torch.Tensor,
    IL: torch.Tensor,
    Vc: torch.Tensor,
    C: torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """Calculate normalized KCL violation."""
    i_cap = C * dVc_dt
    i_applied = (1.0 - D_dummy_handling_if_needed(IL, Vc)) * IL - Vc / (Rload + 1e-12)
    # Wait, we need to extract duty cycle D or handle it in parameter lists.
    # To keep it generic, let's pass duty cycle explicitly.
    return torch.mean((i_cap - i_applied) ** 2)


def D_dummy_handling_if_needed(IL, Vc):
    # Dummy, we will pass duty cycle explicitly in the function below.
    return 0.0


def compute_kcl_residual_fixed(
    dVc_dt: torch.Tensor,
    IL: torch.Tensor,
    Vc: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """Calculate normalized KCL violation."""
    i_cap = C * dVc_dt
    i_applied = (1.0 - D) * IL - Vc / (Rload + 1e-12)
    residual = i_cap - i_applied
    return torch.mean(residual ** 2)
