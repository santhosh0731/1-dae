"""
DAE Formulation for Boost Converter
=====================================
Explicit index-1 DAE: F(x, dx/dt, z) = 0

Differential Variables:  x  = [IL(t), Vc(t)]
Algebraic Variables:     z  = [Vout]

Equations:
  f1: L * dIL/dt - (Vin - (1-D)*Vout) = 0      [KVL / Inductor ODE]
  f2: C * dVc/dt - ((1-D)*IL - Vc/Rload) = 0   [KCL / Capacitor ODE]
  g1: Vout - Vc = 0                              [Algebraic constraint]

Residual vector: F = [f1, f2, g1]^T = 0
"""

import torch
from typing import Dict, Tuple


def dae_residuals(
    IL:     torch.Tensor,  # (B,)  inductor current
    Vc:     torch.Tensor,  # (B,)  capacitor voltage
    Vout:   torch.Tensor,  # (B,)  output voltage (algebraic)
    dIL_dt: torch.Tensor,  # (B,)  d(IL)/dt — predicted or autograd
    dVc_dt: torch.Tensor,  # (B,)  d(Vc)/dt — predicted or autograd
    Vin:    torch.Tensor,  # (B,)  input voltage
    D:      torch.Tensor,  # (B,)  duty cycle
    L:      torch.Tensor,  # (B,)  inductance
    C:      torch.Tensor,  # (B,)  capacitance
    Rload:  torch.Tensor,  # (B,)  load resistance
) -> Dict[str, torch.Tensor]:
    """
    Compute DAE residuals F(x, dx/dt, z) = 0.

    Returns dict with:
      f1     : KVL/Inductor ODE residual
      f2     : KCL/Capacitor ODE residual
      g1     : Algebraic constraint residual
      F_norm : L2 norm of full residual vector
    """
    # f1: L * dIL/dt = Vin - (1-D)*Vout
    f1 = L * dIL_dt - (Vin - (1.0 - D) * Vout)

    # f2: C * dVc/dt = (1-D)*IL - Vc/Rload
    f2 = C * dVc_dt - ((1.0 - D) * IL - Vc / Rload)

    # g1: Vout = Vc  (algebraic: capacitor voltage IS output voltage)
    g1 = Vout - Vc

    # Full residual vector norm
    F_vec  = torch.stack([f1, f2, g1], dim=-1)          # (B, 3)
    F_norm = torch.norm(F_vec, dim=-1).mean()            # scalar

    return {
        'f1':     f1,
        'f2':     f2,
        'g1':     g1,
        'F_norm': F_norm,
        'F_vec':  F_vec,
    }


def steady_state_dae(
    Vin: float, D: float, Rload: float
) -> Tuple[float, float]:
    """
    Analytical steady-state solution for the averaged boost converter DAE.

    At steady state: dIL/dt = 0, dVc/dt = 0
      Vout_ss = Vin / (1 - D)
      IL_ss   = Vout_ss / ((1-D) * Rload)

    Returns:
      (IL_ss, Vout_ss)
    """
    Vout_ss = Vin / (1.0 - D)
    IL_ss   = Vout_ss / ((1.0 - D) * Rload)
    return IL_ss, Vout_ss


def jacobian_dae(
    D: torch.Tensor,
    L: torch.Tensor,
    C: torch.Tensor,
    Rload: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    DAE Jacobian matrices for index analysis and Phase 5 IRK embedding.

    State matrix A (linearised around operating point):
      dx/dt = A*x + B*u
      0     = C*x + D*z

    Returns:
      A: system matrix [2x2]
      B: input matrix  [2x1]
    """
    # Averaged state-space matrices
    # [dIL/dt]   [0,         -(1-D)/L] [IL]   [1/L] [Vin]
    # [dVc/dt] = [(1-D)/C,  -1/(RC)  ] [Vc] + [0  ]
    zero = torch.zeros_like(D)
    A11 = zero
    A12 = -(1.0 - D) / L
    A21 = (1.0 - D) / C
    A22 = -1.0 / (Rload * C)

    A = torch.stack([
        torch.stack([A11, A12], dim=-1),
        torch.stack([A21, A22], dim=-1),
    ], dim=-2)  # (B, 2, 2)

    B = torch.stack([1.0 / L, zero], dim=-1).unsqueeze(-1)  # (B, 2, 1)

    return {'A': A, 'B': B}


if __name__ == "__main__":
    # Quick sanity check
    B = 4
    IL    = torch.tensor([10.0, 20.0, 15.0, 12.0])
    Vc    = torch.tensor([48.0, 60.0, 55.0, 50.0])
    Vout  = Vc.clone()
    dIL   = torch.zeros(B)
    dVc   = torch.zeros(B)
    Vin   = torch.tensor([36.0, 36.0, 36.0, 36.0])
    D     = torch.tensor([0.5,  0.6,  0.55, 0.52])
    L     = torch.full((B,), 50e-6)
    C     = torch.full((B,), 47e-6)
    Rload = torch.ones(B)

    res = dae_residuals(IL, Vc, Vout, dIL, dVc, Vin, D, L, C, Rload)
    print("DAE Residuals:")
    for k, v in res.items():
        if k != 'F_vec':
            print(f"  {k}: {v}")

    IL_ss, Vout_ss = steady_state_dae(36.0, 0.5, 1.0)
    print(f"\nSteady-state: IL={IL_ss:.2f} A, Vout={Vout_ss:.2f} V")
