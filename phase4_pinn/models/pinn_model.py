"""
PINN Model Architecture
========================
7-input, 5-output fully-connected network with Tanh activations.

Inputs  (7): [t, Vin, D, Fs, L, C, Rload]  — all normalized
Outputs (5): [Vout, IL, Vc, dIL_dt, dVc_dt]

Design decisions:
  - Tanh activations: infinitely differentiable → stable autograd physics gradients
  - 5 outputs: explicit derivative prediction avoids repeated autograd in evaluation
  - Output scaling: maps network output back to physical ranges
  - MC Dropout: enabled during uncertainty estimation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network for Boost Converter.

    Architecture:
      Input(7) → [256 → 256 → 256 → 128] Tanh → Output(5)

    Outputs (denormalized):
      [0] Vout(t)   — output voltage     [V]
      [1] IL(t)     — inductor current   [A]
      [2] Vc(t)     — capacitor voltage  [V]
      [3] dIL_dt    — d(IL)/dt           [A/s]
      [4] dVc_dt    — d(Vc)/dt           [V/s]
    """

    OUTPUT_NAMES = ['Vout', 'IL', 'Vc', 'dIL_dt', 'dVc_dt']

    def __init__(
        self,
        input_dim:    int = 7,
        output_dim:   int = 5,
        hidden_dims:  Tuple[int, ...] = (256, 256, 256, 128),
        dropout_rate: float = 0.05,
        # Physical output bounds for sigmoid-based scaling
        Vout_range:   Tuple[float, float] = (0.0, 300.0),
        IL_range:     Tuple[float, float] = (0.0, 1500.0),
        dIL_range:    Tuple[float, float] = (-1e7, 1e7),
        dVc_range:    Tuple[float, float] = (-1e6, 1e6),
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.dropout_rate = dropout_rate

        # ── Network layers ─────────────────────────────────────────
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.Tanh(),
                nn.Dropout(p=dropout_rate),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # ── Output scaling bounds ──────────────────────────────────
        # Stored as buffers (not parameters) → saved with state_dict
        self.register_buffer('Vout_lo', torch.tensor(Vout_range[0]))
        self.register_buffer('Vout_hi', torch.tensor(Vout_range[1]))
        self.register_buffer('IL_lo',   torch.tensor(IL_range[0]))
        self.register_buffer('IL_hi',   torch.tensor(IL_range[1]))
        self.register_buffer('dIL_lo',  torch.tensor(dIL_range[0]))
        self.register_buffer('dIL_hi',  torch.tensor(dIL_range[1]))
        self.register_buffer('dVc_lo',  torch.tensor(dVc_range[0]))
        self.register_buffer('dVc_hi',  torch.tensor(dVc_range[1]))

        # Weight initialization (Xavier for Tanh)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _scale_output(self, raw: torch.Tensor) -> torch.Tensor:
        """Scale raw network output to physical ranges using sigmoid."""
        sig = torch.sigmoid(raw)

        Vout   = self.Vout_lo + sig[:, 0] * (self.Vout_hi - self.Vout_lo)
        IL     = self.IL_lo   + sig[:, 1] * (self.IL_hi   - self.IL_lo)
        Vc     = self.Vout_lo + sig[:, 2] * (self.Vout_hi - self.Vout_lo)
        dIL_dt = self.dIL_lo  + sig[:, 3] * (self.dIL_hi  - self.dIL_lo)
        dVc_dt = self.dVc_lo  + sig[:, 4] * (self.dVc_hi  - self.dVc_lo)

        return torch.stack([Vout, IL, Vc, dIL_dt, dVc_dt], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7)  normalized inputs [t, Vin, D, Fs, L, C, Rload]
        Returns:
            out: (B, 5) [Vout, IL, Vc, dIL_dt, dVc_dt]
        """
        raw = self.net(x)
        return self._scale_output(raw)

    def predict_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning named outputs."""
        out = self.forward(x)
        return {name: out[:, i] for i, name in enumerate(self.OUTPUT_NAMES)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def mc_predict(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo dropout inference for uncertainty estimation.
        Enables dropout during inference to get epistemic uncertainty.

        Returns:
            mean: (B, 5) mean predictions
            std:  (B, 5) standard deviation (uncertainty)
        """
        self.train()  # enable dropout
        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)  # (N, B, 5)
        self.eval()
        return {
            'mean': preds.mean(0),   # (B, 5)
            'std':  preds.std(0),    # (B, 5)
            'preds': preds,
        }


class PINNInputNormalizer(nn.Module):
    """
    Normalizes the 7 PINN inputs [t, Vin, D, Fs, L, C, Rload]
    using precomputed mean and std. Stored as buffers for export.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std + 1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std + self.mean


def build_pinn(config: dict) -> PINNModel:
    """Build PINN from config dict."""
    return PINNModel(
        input_dim    = config['model']['input_dim'],
        output_dim   = config['model']['output_dim'],
        hidden_dims  = tuple(config['model']['hidden_dims']),
        dropout_rate = config['model'].get('dropout_rate', 0.05),
    )


if __name__ == "__main__":
    model = PINNModel()
    print(f"PINN Model — Parameters: {model.count_parameters():,}")
    print(f"Outputs: {model.OUTPUT_NAMES}")
    x = torch.randn(8, 7)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape} = {list(model.OUTPUT_NAMES)}")
    mc = model.mc_predict(x, n_samples=20)
    print(f"MC mean: {mc['mean'].shape}, MC std: {mc['std'].shape}")
