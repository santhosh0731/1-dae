"""
Algebraic Constraints
======================
Specific physical checks for duty cycles, switching limits, output voltage matching,
and power boundaries of the boost converter.
"""

import numpy as np
import torch
from typing import Dict


def check_algebraic_constraints(
    IL: torch.Tensor,
    Vc: torch.Tensor,
    Vout: torch.Tensor,
    Vin: torch.Tensor,
    D: torch.Tensor,
    Rload: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Calculate algebraic violations for the converter:
      1. Vout - Vc = 0 (DAE index-1 algebraic condition)
      2. Pin - Pout = 0 (Power conservation)
    """
    # 1. Output matching constraint
    res_vout_matching = Vout - Vc

    # 2. Power conservation constraint (averaged over switching cycle)
    P_in = Vin * IL * D
    P_out = (Vout ** 2) / (Rload + 1e-12)
    res_power = P_in - P_out

    return {
        'vout_matching': res_vout_matching,
        'power_balance': res_power,
    }
