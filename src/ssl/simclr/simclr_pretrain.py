"""
Self-Supervised Pretraining (SimCLR-style for Circuit Waveforms)
================================================================
Circuit-aware augmentations and NT-Xent loss to pretrain waveform encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class BoostConverterAugmentor:
    """
    Physics-consistent augmentations for self-supervised contrastive learning.
    """
    def __init__(
        self,
        time_shift_frac:  float = 0.1,
        duty_perturb:     float = 0.05,
        noise_std:        float = 0.02,
        freq_scale_range: Tuple = (0.8, 1.2),
    ):
        self.time_shift_frac  = time_shift_frac
        self.duty_perturb     = duty_perturb
        self.noise_std        = noise_std
        self.freq_scale_range = freq_scale_range

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        shift = int(self.time_shift_frac * x.shape[-1] * np.random.uniform(0, 1))
        return torch.roll(x, shift, dims=-1)

    def additive_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise_std * torch.randn_like(x) * x.std()

    def frequency_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(*self.freq_scale_range)
        orig_len = x.shape[-1]
        new_len = int(orig_len * scale)
        x_scaled = F.interpolate(
            x.unsqueeze(0) if x.dim() == 2 else x,
            size=new_len, mode='linear', align_corners=False)
        if new_len > orig_len:
            return x_scaled[..., :orig_len]
        else:
            return F.pad(x_scaled, (0, orig_len - new_len))

    def cutout(self, x: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
        T = x.shape[-1]
        start = int(np.random.uniform(0, T * (1 - frac)))
        length = int(T * frac)
        x = x.clone()
        x[..., start:start + length] = 0.0
        return x

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        augs = [self.time_shift, self.additive_noise, self.cutout]
        chosen = np.random.choice(len(augs), 2, replace=False)
        for idx in chosen:
            x = augs[idx](x)
        return x

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss (SimCLR).
    """
    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.T = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=-1)

        sim = torch.mm(z, z.T) / self.T

        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        pos_idx = torch.arange(B, device=z.device)
        pos_sim = torch.cat([
            sim[pos_idx, pos_idx + B],
            sim[pos_idx + B, pos_idx]
        ], dim=0)

        loss = -pos_sim + torch.logsumexp(sim, dim=-1)
        return loss.mean()

class SSLWaveformEncoder(nn.Module):
    """
    1D ResNet backbone + projection head for SimCLR pretraining.
    """
    def __init__(
        self,
        in_channels: int = 2,
        backbone_dim: int = 256,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            *self._resblocks(64, 128, 2),
            *self._resblocks(128, 256, 2),
            *self._resblocks(256, backbone_dim, 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim), nn.BatchNorm1d(backbone_dim), nn.ReLU(),
            nn.Linear(backbone_dim, proj_dim),
        )

    @staticmethod
    def _resblocks(in_c: int, out_c: int, n: int) -> List:
        blocks = []
        for i in range(n):
            stride = 2 if i == 0 else 1
            inc = in_c if i == 0 else out_c
            blocks.append(ResBlock1d(inc, out_c, stride))
        return blocks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.projector(h)

class ResBlock1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
        )
        self.skip = (nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_c))
            if in_c != out_c or stride != 1 else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

def pretrain_ssl(
    encoder:   SSLWaveformEncoder,
    waveforms: torch.Tensor,
    n_epochs:  int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    temperature: float = 0.07,
    device: str = "cpu",
) -> SSLWaveformEncoder:
    encoder = encoder.to(device)
    augmentor = BoostConverterAugmentor()
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-6)

    N = waveforms.shape[0]
    for epoch in range(n_epochs):
        encoder.train()
        idx = torch.randperm(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = waveforms[idx[start:end]].to(device)
            if batch.shape[0] <= 1:
                continue

            view1 = torch.stack([augmentor.augment(x) for x in batch])
            view2 = torch.stack([augmentor.augment(x) for x in batch])

            z1 = encoder(view1)
            z2 = encoder(view2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"[SSL] Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.4f}")

    return encoder
