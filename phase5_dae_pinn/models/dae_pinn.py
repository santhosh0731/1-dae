"""
DAE-PINN Architecture
======================
8-layer Deep DAE-PINN Model with Custom Sigmoidal Physical Scaling Boundaries.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DAEPINNModel(nn.Module):
    """
    DAE-PINN Architecture with 7-inputs and 5-outputs.
    Enforces physical boundaries on predictions using sigmoid output scaling.
    """

    OUTPUT_NAMES = ['Vout', 'IL', 'Vc', 'dIL_dt', 'dVc_dt']

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256, 128]

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.Tanh())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_d = h_d

        final_layer = nn.Linear(in_d, output_dim)
        nn.init.xavier_uniform_(final_layer.weight, gain=1.0)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

        # Physical boundary scales aligned to dataset sweeps
        self.register_buffer('Vout_lo',  torch.tensor(0.0))
        self.register_buffer('Vout_hi',  torch.tensor(400.0))
        self.register_buffer('IL_lo',    torch.tensor(0.0))
        self.register_buffer('IL_hi',    torch.tensor(1500.0))
        self.register_buffer('Vc_lo',    torch.tensor(0.0))
        self.register_buffer('Vc_hi',    torch.tensor(400.0))
        self.register_buffer('dIL_lo',   torch.tensor(-5e6))
        self.register_buffer('dIL_hi',   torch.tensor(5e6))
        self.register_buffer('dVc_lo',   torch.tensor(-1e7))
        self.register_buffer('dVc_hi',   torch.tensor(3e7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.network(x)

        # Scale outputs into valid physical domains
        Vout   = self.Vout_lo + (self.Vout_hi - self.Vout_lo) * torch.sigmoid(raw[:, 0])
        IL     = self.IL_lo   + (self.IL_hi - self.IL_lo)     * torch.sigmoid(raw[:, 1])
        Vc     = self.Vc_lo   + (self.Vc_hi - self.Vc_lo)     * torch.sigmoid(raw[:, 2])
        dIL_dt = self.dIL_lo  + (self.dIL_hi - self.dIL_lo)   * torch.sigmoid(raw[:, 3])
        dVc_dt = self.dVc_lo  + (self.dVc_hi - self.dVc_lo)   * torch.sigmoid(raw[:, 4])

        return torch.stack([Vout, IL, Vc, dIL_dt, dVc_dt], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Activate dropout during inference for Monte Carlo uncertainty analysis."""
        self.train()  # Keep dropout active
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                preds.append(self.forward(x))
        self.eval()
        stacked = torch.stack(preds, dim=0)  # (N_samples, B, 5)
        return {'preds': stacked}
