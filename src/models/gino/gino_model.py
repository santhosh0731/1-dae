"""
Geometry-Informed Neural Operator (GINO / Geometry-Informed FNO)
=================================================================
Handles non-uniform switching times, implements switching-event positional encoding,
and multi-resolution spectral paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT → complex spectral weight multiply → iFFT.
    Only keeps lowest `modes` frequency components.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        W = torch.complex(self.weights_real, self.weights_imag)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bim,iom->bom', x_ft[:, :, :self.modes], W)

        return torch.fft.irfft(out_ft, n=L, dim=-1)

class GINOFourierBlock(nn.Module):
    """
    Single GINO block: spectral conv (global) + pointwise MLP (local) + residual.
    """
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local    = nn.Conv1d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm1d(width, affine=True)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.local(x))) + x

class GeometryInformedFNO(nn.Module):
    """
    GINO: Geometry-Informed Neural Operator for converter waveforms.
    """
    def __init__(
        self,
        param_dim:    int = 6,
        T:            int = 512,
        modes:        int = 64,
        width:        int = 128,
        n_layers:     int = 5,
        n_signals:    int = 2,
    ):
        super().__init__()
        self.T     = T
        self.width = width
        self.modes = modes

        # Lift: concatenated [params, t, event] -> width
        # Concatenation: [params(param_dim), t(1), event(1)] = param_dim + 2
        self.lift = nn.Linear(param_dim + 2, width)

        self.blocks_coarse = nn.ModuleList([
            GINOFourierBlock(width, modes // 2) for _ in range(n_layers // 2)
        ])
        self.blocks_fine = nn.ModuleList([
            GINOFourierBlock(width, modes) for _ in range(n_layers - n_layers // 2)
        ])

        self.freq_gate = nn.Sequential(
            nn.Linear(modes + 1, modes), nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv1d(width, width // 2, 1), nn.GELU(),
            nn.Conv1d(width // 2, n_signals, 1),
        )

        self.register_buffer('Vout_max', torch.tensor(400.0))
        self.register_buffer('IL_max',   torch.tensor(1500.0))

    def switching_event_encoding(
        self,
        t_norm: torch.Tensor,
        D:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes proximity to switching transitions as a soft bump function.
        """
        t_mod   = t_norm % 1.0
        D_       = D.unsqueeze(1).expand_as(t_mod)
        bump_on  = torch.exp(-50 * (t_mod) ** 2)
        bump_off = torch.exp(-50 * (t_mod - D_) ** 2)
        return (bump_on + bump_off).unsqueeze(-1)

    def forward(
        self,
        x_params: torch.Tensor,
        t_grid:   torch.Tensor,
    ) -> torch.Tensor:
        B = x_params.shape[0]
        T = self.T

        if t_grid.dim() == 1:
            t_grid = t_grid.unsqueeze(0).expand(B, -1)

        D_param = x_params[:, 1]

        event_enc = self.switching_event_encoding(t_grid, D_param)

        p_exp = x_params.unsqueeze(1).expand(B, T, -1)
        t_in  = t_grid.unsqueeze(-1)

        h = torch.cat([p_exp, t_in, event_enc], dim=-1)
        h = self.lift(h)
        h = h.transpose(1, 2)

        for blk in self.blocks_coarse:
            h = blk(h)
        for blk in self.blocks_fine:
            h = blk(h)

        out = self.proj(h)
        out = out.transpose(1, 2)

        Vout = self.Vout_max * torch.sigmoid(out[..., 0])
        IL   = self.IL_max   * torch.sigmoid(out[..., 1])

        return torch.stack([Vout, IL], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
