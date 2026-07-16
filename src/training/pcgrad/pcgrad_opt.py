"""
PCGrad (Projecting Conflicting Gradients)
===========================================
Alleviates multi-task gradient interference by projecting conflicting gradients
onto each other's normal planes.
"""

import torch
import torch.nn.functional as F
from typing import List

class PCGrad:
    """
    Projecting Conflicting Gradients (PCGrad).
    Surgically projects conflicting gradients to minimize gradient conflict.
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._task_grads: List[List[torch.Tensor]] = []

    def zero_grad(self):
        self.optimizer.zero_grad()

    def pc_backward(self, losses: List[torch.Tensor]):
        """
        losses: list of scalar tensors (one per task/loss-term)
        """
        params = [p for group in self.optimizer.param_groups
                  for p in group['params'] if p.requires_grad]

        # Collect per-task gradients
        task_grads = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = [p.grad.clone() if p.grad is not None
                     else torch.zeros_like(p)
                     for p in params]
            task_grads.append(grads)

        # Project conflicting gradients
        proj_grads = [list(g) for g in task_grads]
        n_tasks = len(task_grads)

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                for k, (gi, gj) in enumerate(zip(task_grads[i], task_grads[j])):
                    gi_flat = gi.flatten()
                    gj_flat = gj.flatten()
                    cos_sim = F.cosine_similarity(
                        gi_flat.unsqueeze(0), gj_flat.unsqueeze(0)).item()
                    if cos_sim < 0:
                        # Project out conflicting component
                        proj = (gi_flat.dot(gj_flat) /
                                (gj_flat.dot(gj_flat) + 1e-8))
                        proj_grads[i][k] = (gi - (proj * gj)).reshape(gi.shape)

        # Sum projected gradients
        self.optimizer.zero_grad()
        for k, p in enumerate(params):
            merged = sum(pg[k] for pg in proj_grads)
            p.grad = merged

    def step(self):
        self.optimizer.step()
