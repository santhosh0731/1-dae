"""
Adaptive Learning - Continual Learning
======================================
Prevents catastrophic forgetting using parameter anchoring / L2 distance penalty
relative to baseline models during online updates.
"""

import torch
import torch.nn as nn

class WeightAnchorRegularizer:
    """Computes L2 weight anchoring loss to penalize drift from base weights."""
    def __init__(self, base_model: nn.Module, lambda_anchor: float = 0.05):
        self.base_params = {name: param.clone().detach() for name, param in base_model.named_parameters()}
        self.lambda_anchor = lambda_anchor

    def penalty(self, adapted_model: nn.Module) -> torch.Tensor:
        """Returns the L2 weight distance penalty tensor."""
        loss_penalty = torch.tensor(0.0, device=next(adapted_model.parameters()).device)
        for name, param in adapted_model.named_parameters():
            if name in self.base_params and param.requires_grad:
                loss_penalty += torch.sum((param - self.base_params[name]) ** 2)
        return self.lambda_anchor * loss_penalty
