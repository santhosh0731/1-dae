"""
phase45_upgrade_meta_pinn.py
=============================
REPLACES: Standard DAE-PINN + NTK loss balancing
WITH:      Meta-Learning DAE-PINN (MAML + PINN)
           + Evidential Deep Learning for calibrated uncertainty

WHY HIGHER-LEVEL:
  - Standard PINN must retrain from scratch for each new operating point.
  - MAML-PINN learns an initialization that adapts to new converter
    configurations in O(5-10) gradient steps — critical for real-time
    digital twin adaptation.
  - Evidential DL replaces MC Dropout: predicts Dirichlet/NIG distribution
    parameters directly, giving epistemic + aleatoric uncertainty without
    multiple forward passes at inference time.
  - Second-order meta-gradient captures how physics loss curvature changes
    across operating conditions — the optimizer "knows" the DAE structure.

Inputs:  [t, Vin, D, L, C, Rload]  (7-dim, same as phase5_dae_pinn)
Outputs: [Vout, IL, Vc, dIL_dt, dVc_dt] + uncertainty estimates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from typing import Dict, List, Tuple, Optional
import copy


# ─── Evidential Output Head ───────────────────────────────────────────────────
class EvidentialHead(nn.Module):
    """
    Normal-Inverse-Gamma (NIG) evidential regression head.
    Predicts (γ, ν, α, β) per output — from which we recover:
      - Predictive mean:   γ
      - Epistemic var:     β / (ν(α-1))   ← uncertainty from lack of data
      - Aleatoric var:     β / (α-1)       ← uncertainty from noise
    No MC sampling at test time; closed-form uncertainty.
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
        nu    = F.softplus(self.nu_head(h))    + 1e-6  # > 0
        alpha = F.softplus(self.alpha_head(h)) + 1.0   # > 1
        beta  = F.softplus(self.beta_head(h))  + 1e-6  # > 0
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
        """Evidence regularization: penalize high evidence for wrong predictions."""
        err = (y - gamma).abs()
        reg = err * (2.0 * nu + alpha)
        return reg.mean()


# ─── Meta-DAE-PINN Model ─────────────────────────────────────────────────────
class MetaDAEPINN(nn.Module):
    """
    MAML-compatible DAE-PINN.

    The inner network is identical to DAEPINNModel (Phase 5), but the
    forward() accepts fast_weights for MAML inner-loop adaptation.
    The evidential head is added on top for uncertainty quantification.

    Task definition: each (Vin, D, L, C, Rload) configuration = one task.
    Support set: a few collocation points from that configuration.
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

        # Shared feature trunk
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [nn.Linear(in_d, h_d), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_d = h_d
        self.trunk = nn.Sequential(*layers)
        self.trunk_out_dim = in_d

        # Evidential head
        self.evid_head = EvidentialHead(in_d, n_outputs=n_outputs)

        # Physical boundary buffers
        self.register_buffer('out_lo', torch.tensor([  0.0,    0.0,    0.0, -5e6, -1e7]))
        self.register_buffer('out_hi', torch.tensor([400.0, 1500.0, 400.0,  5e6,  3e7]))

    def forward(
        self,
        x: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (point_pred, gamma, nu, alpha, beta).
        point_pred = γ scaled to physical domain.
        fast_weights: if provided, used instead of self.trunk weights
                      during MAML inner loop.
        """
        if fast_weights is None:
            h = self.trunk(x)
        else:
            # Manual forward using fast_weights (for MAML inner loop)
            h = x
            keys = list(fast_weights.keys())
            weight_keys = [k for k in keys if 'weight' in k and 'trunk' in k]
            bias_keys   = [k for k in keys if 'bias'   in k and 'trunk' in k]
            for wk, bk in zip(weight_keys, bias_keys):
                h = F.linear(h, fast_weights[wk], fast_weights[bk])
                h = torch.tanh(h)

        gamma, nu, alpha, beta = self.evid_head(h)

        # Scale γ (mean prediction) to physical domain
        point_pred = (self.out_lo +
                      (self.out_hi - self.out_lo) * torch.sigmoid(gamma))

        return point_pred, gamma, nu, alpha, beta

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward → mean + epistemic + aleatoric uncertainty.
        No MC sampling required.
        """
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


# ─── MAML Trainer ────────────────────────────────────────────────────────────
class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) training loop for Meta-DAE-PINN.

    Outer loop: updates θ via meta-gradient across tasks
    Inner loop: fast adaptation to each task's physics constraints

    Physics loss is applied in BOTH inner and outer loops, ensuring the
    meta-initialization already satisfies DAE constraints.
    """

    def __init__(
        self,
        model:          MetaDAEPINN,
        inner_lr:       float = 1e-3,
        outer_lr:       float = 1e-4,
        n_inner_steps:  int = 5,
        first_order:    bool = False,   # FOMAML (True) or full MAML (False)
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

    # ─ Physics residuals (inline, no external dep for portability) ──────────
    @staticmethod
    def physics_loss(
        pred:   torch.Tensor,   # (B, 5) [Vout, IL, Vc, dIL/dt, dVc/dt]
        params: torch.Tensor,   # (B, 5) [Vin, D, L, C, Rload]
    ) -> torch.Tensor:
        Vout, IL, Vc, dIL, dVc = pred.unbind(1)
        Vin, D, L, C, Rload    = params.unbind(1)

        V_REF = 114.5; I_REF = 380.0; P_REF = 43510.0

        kvl = ((L * dIL - Vin * D + Vout * (1 - D)) / V_REF) ** 2
        kcl = ((C * dVc - IL * (1 - D) + Vout / Rload) / I_REF) ** 2
        dae = ((Vout - Vc) / V_REF) ** 2
        pwr = ((Vin * IL - Vout ** 2 / Rload) / P_REF) ** 2

        return kvl.mean() + kcl.mean() + dae.mean() + pwr.mean()

    # ─ Inner loop: fast adaptation ──────────────────────────────────────────
    def inner_loop(
        self,
        support_x:      torch.Tensor,   # (K, 7) support inputs
        support_y:      torch.Tensor,   # (K, 5) support targets
        phys_params:    torch.Tensor,   # (K, 5) physics params
    ) -> Dict[str, torch.Tensor]:
        """Returns fast_weights after n_inner_steps gradient steps."""
        fast_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for step in range(self.n_inner_steps):
            pred, gamma, nu, alpha, beta = self.model(support_x, fast_weights)

            # Combined loss
            data_nll = EvidentialHead.nig_nll(support_y, gamma, nu, alpha, beta)
            data_reg  = EvidentialHead.nig_reg(support_y, gamma, nu, alpha)
            phys      = self.physics_loss(pred, phys_params)

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

    # ─ Outer meta-update ────────────────────────────────────────────────────
    def meta_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
    ) -> float:
        """
        tasks: list of dicts with keys
               'support_x', 'support_y', 'support_phys',
               'query_x',   'query_y',   'query_phys'
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for task in tasks:
            # Inner loop on support set
            fast_weights = self.inner_loop(
                task['support_x'],
                task['support_y'],
                task['support_phys'],
            )

            # Outer loop: evaluate on query set using fast_weights
            pred_q, gamma_q, nu_q, alpha_q, beta_q = self.model(
                task['query_x'], fast_weights)

            data_nll = EvidentialHead.nig_nll(
                task['query_y'], gamma_q, nu_q, alpha_q, beta_q)
            data_reg  = EvidentialHead.nig_reg(
                task['query_y'], gamma_q, nu_q, alpha_q)
            phys = self.physics_loss(pred_q, task['query_phys'])

            task_loss = data_nll + self.evid_weight * data_reg + self.phys_weight * phys
            meta_loss = meta_loss + task_loss / len(tasks)

        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()
        self.meta_scheduler.step()

        return float(meta_loss)

    # ─ Fast adapt at test time ───────────────────────────────────────────────
    def adapt(
        self,
        new_x:       torch.Tensor,   # Few-shot support from new converter config
        new_y:       torch.Tensor,
        new_phys:    torch.Tensor,
        n_adapt:     int = 10,
    ) -> MetaDAEPINN:
        """
        Clone model and adapt to a new configuration in n_adapt steps.
        Does NOT modify the original meta-parameters.
        """
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


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cpu")

    model = MetaDAEPINN(input_dim=7, hidden_dims=[256, 256, 128], n_outputs=5)
    trainer = MAMLTrainer(model, n_inner_steps=3, first_order=True)

    # Fake tasks
    def make_task(B=16, K=8):
        return {
            'support_x':    torch.randn(K, 7),
            'support_y':    torch.randn(K, 5),
            'support_phys': torch.rand(K, 5).abs() + 0.1,
            'query_x':      torch.randn(B, 7),
            'query_y':      torch.randn(B, 5),
            'query_phys':   torch.rand(B, 5).abs() + 0.1,
        }

    tasks = [make_task() for _ in range(4)]
    loss = trainer.meta_step(tasks)
    print(f"[MAML] Meta loss: {loss:.4f}")

    # Test uncertainty
    x_test = torch.randn(10, 7)
    unc = model.predict_with_uncertainty(x_test)
    print(f"[Evidential] Mean: {unc['mean'].shape}, "
          f"Epistemic: {unc['epistemic'].mean():.4f}, "
          f"Aleatoric: {unc['aleatoric'].mean():.4f}")

    # Test fast adaptation
    new_x = torch.randn(5, 7)
    new_y = torch.randn(5, 5)
    new_p = torch.rand(5, 5).abs() + 0.1
    adapted = trainer.adapt(new_x, new_y, new_p, n_adapt=5)
    print(f"[Adapt] Adapted model params: "
          f"{sum(p.numel() for p in adapted.parameters()):,}")
