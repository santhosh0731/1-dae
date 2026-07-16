"""
Physics-Aware Mamba State Space Model (SSM)
===========================================
Replaces 1D-CNN and TCN with selective state space models and soft-differentiable
physics constraint layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SelectiveScanSSM(nn.Module):
    """
    Mamba-style Selective State Space Model (S6).
    Input-dependent (B, C, Δ) matrices enable content-aware sequence modeling.
    """
    def __init__(
        self,
        d_model:  int = 128,
        d_state:  int = 16,
        d_conv:   int = 4,
        expand:   int = 2,
        dt_min:   float = 1e-7,
        dt_max:   float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner = int(d_model * expand)

        # Input projection
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Depth-wise conv
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, bias=True)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Δt initialization
        dt_init_std = d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.act = nn.SiLU()

    def ssm(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        d_state = self.d_state

        x_dbl = self.x_proj(u)
        dt_r   = x_dbl[..., :1]
        B_in   = x_dbl[..., 1:d_state+1]
        C_in   = x_dbl[..., d_state+1:]

        dt = F.softplus(self.dt_proj(dt_r))

        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_in.unsqueeze(2)

        h = torch.zeros(B, D, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        return y + u * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)

        x_ = x_.transpose(1, 2)
        x_ = self.conv1d(x_)[..., :L]
        x_ = x_.transpose(1, 2)
        x_ = self.act(x_)

        y = self.ssm(x_)
        y = y * self.act(z)
        return self.out_proj(y)

class BoostPhysicsConstraintLayer(nn.Module):
    """
    Soft-differentiable KVL/KCL enforcement applied per time step.
    Corrects SSM outputs to respect circuit equations at each t.
    """
    def __init__(self, correction_scale: float = 0.1):
        super().__init__()
        self.correction_scale = correction_scale
        self.correction_net = nn.Sequential(
            nn.Linear(9, 32), nn.SiLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(
        self,
        pred: torch.Tensor,
        params: torch.Tensor,
        dt: float = 1e-6,
    ) -> torch.Tensor:
        B, T, _ = pred.shape
        Vin    = params[:, 0:1].unsqueeze(1).expand(B, T, 1)
        D      = params[:, 1:2].unsqueeze(1).expand(B, T, 1)
        L_val  = params[:, 2:3].unsqueeze(1).expand(B, T, 1)
        C_val  = params[:, 3:4].unsqueeze(1).expand(B, T, 1)
        Rload  = params[:, 4:5].unsqueeze(1).expand(B, T, 1)

        Vout = pred[..., 0:1]
        IL   = pred[..., 1:2]

        dIL_physics = (Vin * D - Vout * (1 - D)) / (L_val + 1e-9)
        dVc_physics = (IL * (1 - D) - Vout / (Rload + 1e-9)) / (C_val + 1e-9)

        feat = torch.cat([Vout, IL, Vin, D, L_val, C_val, Rload,
                          dIL_physics, dVc_physics], dim=-1)
        correction = self.correction_net(feat) * self.correction_scale
        return pred + correction

class PhysicsMambaSSM(nn.Module):
    """
    Physics-Aware Mamba SSM Network.
    """
    def __init__(
        self,
        param_dim:  int = 6,
        d_model:    int = 128,
        d_state:    int = 16,
        n_layers:   int = 6,
        T:          int = 512,
        n_signals:  int = 2,
    ):
        super().__init__()
        self.T = T

        self.cond_encoder = nn.Sequential(
            nn.Linear(param_dim, 128), nn.SiLU(),
            nn.Linear(128, d_model),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm':  SelectiveScanSSM(d_model=d_model, d_state=d_state),
                'ff_norm': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.SiLU(),
                    nn.Linear(d_model * 4, d_model),
                )
            })
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_signals)
        self.physics_layer = BoostPhysicsConstraintLayer(correction_scale=0.05)

    def forward(
        self,
        x_params:   torch.Tensor,
        t_grid:     Optional[torch.Tensor] = None,
        phys_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_params.shape[0]
        T = self.T

        if t_grid is None:
            t_grid = torch.linspace(0, 1, T, device=x_params.device).unsqueeze(0).expand(B, -1)
        elif t_grid.dim() == 1:
            t_grid = t_grid.unsqueeze(0).expand(B, -1)

        cond = self.cond_encoder(x_params)
        cond = cond.unsqueeze(1).expand(B, T, -1)

        t_in = t_grid.unsqueeze(-1)
        t_emb = self.time_embed(t_in)

        h = cond + t_emb

        for layer in self.layers:
            h = h + layer['ssm'](layer['norm'](h))
            h = h + layer['ff'](layer['ff_norm'](h))

        h = self.out_norm(h)
        out = self.out_proj(h)

        if phys_params is not None:
            # Check length: if phys_params is 5-dim [Vin, D, L, C, Rload], but we pass param_dim=6 [Vin, D, Fs, L, C, Rload],
            # we need to map or extract the proper physical parameters.
            # Let's check dimensions of phys_params
            if phys_params.shape[1] == 6:
                # [Vin, D, Fs, L, C, Rload] -> [Vin, D, L, C, Rload]
                phys_params_5 = torch.cat([phys_params[:, 0:2], phys_params[:, 3:6]], dim=-1)
            else:
                phys_params_5 = phys_params
            out = self.physics_layer(out, phys_params_5)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_mamba_surrogate(
    param_dim: int = 6,
    T: int = 512,
    d_model: int = 128,
    n_layers: int = 6,
) -> PhysicsMambaSSM:
    return PhysicsMambaSSM(
        param_dim=param_dim,
        d_model=d_model,
        d_state=16,
        n_layers=n_layers,
        T=T,
    )
