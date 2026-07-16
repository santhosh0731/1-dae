"""
Meta-Learning DAE-PINN (MAML + PINN)
=====================================
MAML-compatible DAE-PINN to adapt to new converter configurations in a few gradient steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from typing import Dict, List, Tuple, Optional
import copy
from src.models.evidential.evidential_model import EvidentialHead

class MetaDAEPINN(nn.Module):
    """
    MAML-compatible DAE-PINN with Evidential Head.
    """
    def __init__(
        self,
        input_dim:   int = 7,
        hidden_dims: List[int] = None,
        n_outputs:   int = 5,
        dropout:     float = 0.05,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256, 128]

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [nn.Linear(in_d, h_d), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_d = h_d
        self.trunk = nn.Sequential(*layers)
        self.trunk_out_dim = in_d
        self.evid_head = EvidentialHead(in_d, n_outputs=n_outputs)

        # Scale outputs into physical domain
        self.register_buffer('out_lo', torch.tensor([  0.0,    0.0,    0.0, -5e6, -1e7]))
        self.register_buffer('out_hi', torch.tensor([400.0, 1500.0, 400.0,  5e6,  3e7]))

    def forward(
        self,
        x: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if fast_weights is None:
            h = self.trunk(x)
        else:
            h = x
            keys = list(fast_weights.keys())
            weight_keys = [k for k in keys if 'weight' in k and 'trunk' in k]
            bias_keys   = [k for k in keys if 'bias'   in k and 'trunk' in k]
            for wk, bk in zip(weight_keys, bias_keys):
                h = F.linear(h, fast_weights[wk], fast_weights[bk])
                h = torch.tanh(h)

        gamma, nu, alpha, beta = self.evid_head(h)
        point_pred = (self.out_lo + (self.out_hi - self.out_lo) * torch.sigmoid(gamma))
        return point_pred, gamma, nu, alpha, beta

    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            pred, gamma, nu, alpha, beta = self.forward(x)
        epistemic = beta / (nu * (alpha - 1).clamp(min=1e-6))
        aleatoric = beta / (alpha - 1).clamp(min=1e-6)
        return {
            'mean':      pred,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total_var': epistemic + aleatoric,
        }

class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) training loop for Meta-DAE-PINN.
    """
    def __init__(
        self,
        model:          MetaDAEPINN,
        inner_lr:       float = 1e-3,
        outer_lr:       float = 1e-4,
        n_inner_steps:  int = 5,
        first_order:    bool = False,
        phys_weight:    float = 0.1,
        evid_weight:    float = 0.01,
    ):
        self.model = model
        self.inner_lr      = inner_lr
        self.outer_lr      = outer_lr
        self.n_inner_steps = n_inner_steps
        self.first_order   = first_order
        self.phys_weight   = phys_weight
        self.evid_weight   = evid_weight

        self.meta_optimizer = torch.optim.AdamW(
            model.parameters(), lr=outer_lr, weight_decay=1e-5)
        self.meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=1000, eta_min=1e-6)

    @staticmethod
    def physics_loss(
        pred:   torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        Vout, IL, Vc, dIL, dVc = pred.unbind(1)
        Vin, D, L, C, Rload    = params.unbind(1)

        V_REF = 114.5; I_REF = 380.0; P_REF = 43510.0

        kvl = ((L * dIL - Vin * D + Vout * (1 - D)) / V_REF) ** 2
        kcl = ((C * dVc - IL * (1 - D) + Vout / Rload) / I_REF) ** 2
        dae = ((Vout - Vc) / V_REF) ** 2
        pwr = ((Vin * IL - Vout ** 2 / Rload) / P_REF) ** 2

        return kvl.mean() + kcl.mean() + dae.mean() + pwr.mean()

    def inner_loop(
        self,
        support_x:      torch.Tensor,
        support_y:      torch.Tensor,
        phys_params:    torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fast_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for step in range(self.n_inner_steps):
            pred, gamma, nu, alpha, beta = self.model(support_x, fast_weights)

            data_nll = EvidentialHead.nig_nll(support_y, gamma, nu, alpha, beta)
            data_reg = EvidentialHead.nig_reg(support_y, gamma, nu, alpha)
            phys     = self.physics_loss(pred, phys_params)

            loss = data_nll + self.evid_weight * data_reg + self.phys_weight * phys

            grads = autograd.grad(
                loss, list(fast_weights.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            fast_weights = {
                name: (param - self.inner_lr * (g if g is not None else torch.zeros_like(param)))
                for (name, param), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def meta_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
    ) -> float:
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for task in tasks:
            fast_weights = self.inner_loop(
                task['support_x'],
                task['support_y'],
                task['support_phys'],
            )

            pred_q, gamma_q, nu_q, alpha_q, beta_q = self.model(
                task['query_x'], fast_weights)

            data_nll = EvidentialHead.nig_nll(
                task['query_y'], gamma_q, nu_q, alpha_q, beta_q)
            data_reg = EvidentialHead.nig_reg(
                task['query_y'], gamma_q, nu_q, alpha_q)
            phys = self.physics_loss(pred_q, task['query_phys'])

            task_loss = data_nll + self.evid_weight * data_reg + self.phys_weight * phys
            meta_loss = meta_loss + task_loss / len(tasks)

        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()
        self.meta_scheduler.step()

        return float(meta_loss)

    def adapt(
        self,
        new_x:       torch.Tensor,
        new_y:       torch.Tensor,
        new_phys:    torch.Tensor,
        n_adapt:     int = 10,
    ) -> MetaDAEPINN:
        adapted = copy.deepcopy(self.model)
        opt = torch.optim.Adam(adapted.parameters(), lr=self.inner_lr)

        adapted.train()
        for _ in range(n_adapt):
            opt.zero_grad()
            pred, gamma, nu, alpha, beta = adapted(new_x)
            loss = (EvidentialHead.nig_nll(new_y, gamma, nu, alpha, beta)
                    + self.phys_weight * self.physics_loss(pred, new_phys))
            loss.backward()
            opt.step()

        adapted.eval()
        return adapted
