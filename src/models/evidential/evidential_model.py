"""
Evidential Deep Learning for Calibrated Uncertainty
===================================================
Normal-Inverse-Gamma evidential regression head. Predicts mean, epistemic, and
aleatoric uncertainty in a single forward pass without MC sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EvidentialHead(nn.Module):
    """
    Normal-Inverse-Gamma (NIG) evidential regression head.
    """
    def __init__(self, in_features: int, n_outputs: int = 5):
        super().__init__()
        self.n_outputs = n_outputs
        self.gamma_head = nn.Linear(in_features, n_outputs)
        self.nu_head    = nn.Linear(in_features, n_outputs)
        self.alpha_head = nn.Linear(in_features, n_outputs)
        self.beta_head  = nn.Linear(in_features, n_outputs)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.gamma_head(h)
        nu    = F.softplus(self.nu_head(h))    + 1e-6
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta  = F.softplus(self.beta_head(h))  + 1e-6
        return gamma, nu, alpha, beta

    @staticmethod
    def nig_nll(
        y:     torch.Tensor,
        gamma: torch.Tensor,
        nu:    torch.Tensor,
        alpha: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood under Normal-Inverse-Gamma prior."""
        omega = 2.0 * beta * (1.0 + nu)
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll.mean()

    @staticmethod
    def nig_reg(
        y:     torch.Tensor,
        gamma: torch.Tensor,
        nu:    torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize high evidence for wrong predictions."""
        err = (y - gamma).abs()
        reg = err * (2.0 * nu + alpha)
        return reg.mean()
