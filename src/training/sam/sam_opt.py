"""
Sharpness-Aware Minimization (SAM) Optimizer
=============================================
SAM finds flatter minima in the loss landscape, which enhances generalization
across unseen operating conditions.
"""

import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al. 2021).
    Two-step optimizer: ascent to perturbed weights → descent at flat minimum.
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute gradient ascent step to ε-neighbourhood."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore original weights and apply base optimizer step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'e_w' in self.state[p]:
                    p.sub_(self.state[p]['e_w'])

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.norm(2)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2)

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step.")
