"""
NTK Gradient Weighting Module
==============================
Computes dynamic weights for all physics loss terms based on gradient norm matching.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class NTKGradientBalancing:
    """
    Computes adaptive lambda values so that the gradient norm of each
    physics loss matches the gradient norm of the data loss.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.9,
        update_freq: int = 10,
        min_w: float = 0.01,
        max_w: float = 100.0,
    ):
        self.model = model
        self.alpha = alpha
        self.update_freq = update_freq
        self.min_w = min_w
        self.max_w = max_w
        self.step_count = 0
        self._ema_weights: Dict[str, float] = {}

    def _grad_norm(self, loss: torch.Tensor) -> float:
        """Compute the norm of gradients w.r.t model weights."""
        grads = torch.autograd.grad(
            loss, self.model.parameters(),
            retain_graph=True, create_graph=False, allow_unused=True
        )
        total = 0.0
        for g in grads:
            if g is not None:
                total += g.detach().pow(2).sum().item()
        return float(np.sqrt(total)) + 1e-9

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        data_loss: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute gradient balanced weights."""
        self.step_count += 1

        if self.step_count % self.update_freq != 0 and self._ema_weights:
            return self._ema_weights.copy()

        g_data = self._grad_norm(data_loss)
        new_w = {}
        for k, loss_val in losses.items():
            if loss_val is None or not loss_val.requires_grad:
                new_w[k] = 1.0
                continue

            g_phys = self._grad_norm(loss_val)
            raw = g_data / (g_phys + 1e-9)
            raw = float(np.clip(raw, self.min_w, self.max_w))

            prev = self._ema_weights.get(k, raw)
            new_w[k] = self.alpha * prev + (1.0 - self.alpha) * raw

        self._ema_weights = new_w
        return new_w
