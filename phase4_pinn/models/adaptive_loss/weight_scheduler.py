"""
Adaptive Loss Weight Scheduler
================================
Implements NTK gradient-balancing (Wang et al. 2021) for automatic
lambda weight computation during Phase C of PINN training.

Reference:
  Wang, S., Teng, Y., & Perdikaris, P. (2021).
  "Understanding and mitigating gradient pathologies in physics-informed
  neural networks." SIAM Journal on Scientific Computing.

Algorithm:
  At each step:
    1. Compute gradient of data loss w.r.t. network parameters
    2. Compute gradient of each physics loss w.r.t. network parameters
    3. Scale each physics lambda so its gradient norm matches data gradient norm
    4. Apply EMA smoothing to prevent oscillation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class GradientBalancedWeightScheduler:
    """
    NTK gradient-balanced adaptive loss weight scheduler.

    Usage:
        scheduler = GradientBalancedWeightScheduler(model, alpha=0.9)
        lambdas = scheduler.update(losses_dict, data_loss)
    """

    def __init__(
        self,
        model:      nn.Module,
        alpha:      float = 0.9,    # EMA smoothing factor
        update_freq: int  = 10,     # Update weights every N steps
        min_lambda: float = 0.01,   # Minimum allowed weight
        max_lambda: float = 100.0,  # Maximum allowed weight
    ):
        self.model       = model
        self.alpha       = alpha
        self.update_freq = update_freq
        self.min_lambda  = min_lambda
        self.max_lambda  = max_lambda
        self.step_count  = 0

        # EMA-smoothed weights (initialized to 1.0)
        self._ema_weights: Dict[str, float] = {}

    def _grad_norm(self, loss: torch.Tensor) -> float:
        """Compute L2 norm of gradients of loss w.r.t. model parameters."""
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
        losses:     Dict[str, torch.Tensor],  # {'kvl': t, 'kcl': t, ...}
        data_loss:  torch.Tensor,              # L_data tensor
        verbose:    bool = False,
    ) -> Dict[str, float]:
        """
        Compute gradient-balanced weights for each physics loss.

        Returns updated lambda dict with same keys as `losses`.
        """
        self.step_count += 1

        # Only recompute weights every `update_freq` steps (expensive)
        if self.step_count % self.update_freq != 0:
            return self._ema_weights.copy() if self._ema_weights else \
                   {k: 1.0 for k in losses}

        # Gradient norm of data loss (reference)
        g_data = self._grad_norm(data_loss)

        new_weights = {}
        for key, phys_loss in losses.items():
            if phys_loss is None or not phys_loss.requires_grad:
                new_weights[key] = 1.0
                continue

            g_phys = self._grad_norm(phys_loss)
            raw_lambda = g_data / (g_phys + 1e-9)

            # Clamp to valid range
            raw_lambda = float(np.clip(raw_lambda, self.min_lambda, self.max_lambda))

            # EMA smoothing
            prev = self._ema_weights.get(key, raw_lambda)
            smoothed = self.alpha * prev + (1.0 - self.alpha) * raw_lambda
            new_weights[key] = smoothed

        self._ema_weights = new_weights

        if verbose:
            print("  [Lambda] " + " | ".join(
                f"{k}={v:.3f}" for k, v in new_weights.items()))

        return new_weights


class CurriculumScheduler:
    """
    3-phase curriculum scheduler for PINN training.

    Phase A (data only) → Phase B (gentle physics) → Phase C (adaptive)
    """

    def __init__(self, config: dict):
        self.cfg   = config['curriculum']
        self.ph_a  = self.cfg['phase_a']
        self.ph_b  = self.cfg['phase_b']
        self.ph_c  = self.cfg['phase_c']
        self.adaptive_scheduler: Optional[GradientBalancedWeightScheduler] = None

    def set_adaptive_scheduler(self, scheduler: GradientBalancedWeightScheduler):
        self.adaptive_scheduler = scheduler

    def get_weights(
        self,
        epoch: int,
        losses: Optional[Dict[str, torch.Tensor]] = None,
        data_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Return loss weights for current epoch.
        In Phase C, uses gradient balancing if scheduler is set.
        """
        if epoch <= self.ph_a['end_epoch']:
            # Phase A: data-only warm-start
            return {
                'data': self.ph_a['lambda_data'],
                'kvl':  self.ph_a['lambda_kvl'],
                'kcl':  self.ph_a['lambda_kcl'],
                'dae':  self.ph_a['lambda_dae'],
                'bc':   self.ph_a['lambda_bc'],
                'ic':   self.ph_a['lambda_ic'],
                'pwr':  self.ph_a['lambda_pwr'],
            }
        elif epoch <= self.ph_b['end_epoch']:
            # Phase B: fixed gentle physics
            return {
                'data': self.ph_b['lambda_data'],
                'kvl':  self.ph_b['lambda_kvl'],
                'kcl':  self.ph_b['lambda_kcl'],
                'dae':  self.ph_b['lambda_dae'],
                'bc':   self.ph_b['lambda_bc'],
                'ic':   self.ph_b['lambda_ic'],
                'pwr':  0.0,
            }
        else:
            # Phase C: adaptive gradient balancing
            if self.adaptive_scheduler and losses and data_loss is not None:
                phys_losses = {k: v for k, v in losses.items() if k != 'data'}
                adaptive = self.adaptive_scheduler.update(phys_losses, data_loss)
                return {
                    'data': 1.0,
                    **adaptive,
                    'bc': 0.5,
                    'ic': 0.5,
                }
            else:
                # Fallback to fixed Phase C weights
                return {
                    'data': 1.0,
                    'kvl':  1.0,
                    'kcl':  1.0,
                    'dae':  0.5,
                    'bc':   0.5,
                    'ic':   0.5,
                    'pwr':  0.3,
                }

    def phase_name(self, epoch: int) -> str:
        if epoch <= self.ph_a['end_epoch']:
            return 'Phase A (data-only)'
        elif epoch <= self.ph_b['end_epoch']:
            return 'Phase B (gentle physics)'
        else:
            return 'Phase C (adaptive)'
