"""
POD-DeepONet (Proper Orthogonal Decomposition DeepONet)
========================================================
Trunk net is replaced by fixed/pre-fit POD basis functions computed from
training waveforms via SVD.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class PODDeepONet(nn.Module):
    """
    Proper Orthogonal Decomposition DeepONet.
    Instead of learning trunk net basis from scratch, we use POD modes
    computed from training waveforms as fixed trunk basis functions.
    The branch net only needs to predict coefficients (much lower dim).
    """
    def __init__(
        self,
        param_dim:   int = 6,
        pod_modes:   int = 32,
        n_signals:   int = 2,
        branch_dims: List[int] = None,
    ):
        super().__init__()
        self.pod_modes = pod_modes
        self.n_signals = n_signals

        if branch_dims is None:
            branch_dims = [param_dim, 256, 256, 128, pod_modes * n_signals]

        # Branch net: operating params → POD coefficients
        layers = []
        for i in range(len(branch_dims) - 1):
            layers.append(nn.Linear(branch_dims[i], branch_dims[i + 1]))
            if i < len(branch_dims) - 2:
                layers.append(nn.GELU())
        self.branch_net = nn.Sequential(*layers)

        # POD basis registered as buffers
        self.register_buffer('pod_basis', None)
        self.register_buffer('pod_mean',  None)
        self.is_fitted = False

    def fit_pod(
        self,
        Y_train: np.ndarray,
        n_modes: Optional[int] = None,
    ):
        """
        Compute POD basis from training waveforms via SVD.
        Must be called before forward().
        """
        n_modes = n_modes or self.pod_modes
        N, T, S = Y_train.shape

        bases, means = [], []
        for s in range(S):
            Ys = Y_train[:, :, s]
            mean_s = Ys.mean(axis=0)
            Yc = Ys - mean_s[None, :]
            U, sigma, Vt = np.linalg.svd(Yc, full_matrices=False)
            basis_s = Vt[:n_modes, :].T
            bases.append(basis_s)
            means.append(mean_s)

        basis_tensor = torch.tensor(
            np.stack(bases, axis=-1), dtype=torch.float32)
        mean_tensor  = torch.tensor(
            np.stack(means, axis=-1), dtype=torch.float32)

        self.pod_basis = basis_tensor
        self.pod_mean  = mean_tensor
        self.is_fitted = True

        explained = []
        for s in range(S):
            Ys = Y_train[:, :, s] - Y_train[:, :, s].mean(axis=0)
            _, sigma, _ = np.linalg.svd(Ys, full_matrices=False)
            ev = (sigma[:n_modes] ** 2).sum() / (sigma ** 2).sum()
            explained.append(ev)
        print(f"[POD] Explained variance: "
              + " | ".join(f"Signal {s}: {ev:.4f}" for s, ev in enumerate(explained)))

    def forward(
        self,
        x_params: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_fitted, "Call fit_pod(Y_train) first."

        coeffs = self.branch_net(x_params)
        coeffs = coeffs.view(-1, self.n_signals, self.pod_modes)

        T = self.pod_basis.shape[0]
        basis = self.pod_basis.permute(2, 1, 0)
        recon = torch.einsum('bsm,smt->bts', coeffs, basis)
        recon = recon + self.pod_mean.unsqueeze(0)

        return recon

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
