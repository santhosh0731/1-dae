"""
Deep Kernel Learning (DKL) with SVGP
=====================================
Replaces standard GPR and SVR with a sparse variational GP on top of a
learned deep feature extractor.
"""

import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple

try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        UnwhitenedVariationalStrategy,
    )
    from gpytorch.kernels import (
        ScaleKernel, RBFKernel, MaternKernel, LinearKernel
    )
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.mlls import VariationalELBO
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoostFeatureNet(nn.Module):
    """
    Maps raw 7-dim operating conditions to a 32-dim learned kernel space.
    Uses residual connections + layer-norm for stable DKL training.
    """
    def __init__(self, input_dim: int = 7, feature_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        self.block1 = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64),
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, feature_dim), nn.LayerNorm(feature_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(x))
        h = h + self.block1(h)
        return self.block2(h)

if HAS_GPYTORCH:
    class SVGPModel(ApproximateGP):
        """
        Sparse Variational GP with inducing points in the learned feature space.
        """
        def __init__(self, inducing_points: torch.Tensor):
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0))
            variational_strategy = UnwhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True)
            super().__init__(variational_strategy)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = ScaleKernel(
                MaternKernel(nu=2.5) * RBFKernel() + LinearKernel()
            )

        def forward(self, x: torch.Tensor) -> MultivariateNormal:
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x)
            )
else:
    class SVGPModel(nn.Module):
        def __init__(self, inducing_points):
            super().__init__()
            self.inducing_points = inducing_points
            self.linear = nn.Linear(inducing_points.shape[1], 1)
        def forward(self, x):
            return self.linear(x)

class DeepKernelLearning(nn.Module):
    """
    Multi-output DKL: one shared feature net, one SVGP per output target.
    """
    def __init__(
        self,
        input_dim:    int = 7,
        n_outputs:    int = 5,
        feature_dim:  int = 32,
        n_inducing:   int = 128,
        X_init:       torch.Tensor = None,
    ):
        super().__init__()
        self.feature_net = BoostFeatureNet(input_dim, feature_dim)
        self.has_gpytorch = HAS_GPYTORCH

        if HAS_GPYTORCH:
            if X_init is not None:
                with torch.no_grad():
                    feat_init = self.feature_net(X_init[:n_inducing].float())
            else:
                feat_init = torch.randn(n_inducing, feature_dim)

            self.gps         = nn.ModuleList()
            self.likelihoods = nn.ModuleList()
            for _ in range(n_outputs):
                gp  = SVGPModel(feat_init.clone().detach())
                lik = GaussianLikelihood()
                self.gps.append(gp)
                self.likelihoods.append(lik)
        else:
            self.dummy_heads = nn.ModuleList([
                nn.Linear(feature_dim, 1) for _ in range(n_outputs)
            ])

    def forward(self, x: torch.Tensor, target_idx: int):
        features = self.feature_net(x)
        if self.has_gpytorch:
            return self.gps[target_idx](features)
        else:
            return self.dummy_heads[target_idx](features)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means, stds = [], []
        features = self.feature_net(x)
        if self.has_gpytorch:
            for gp, lik in zip(self.gps, self.likelihoods):
                gp.eval(); lik.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = lik(gp(features))
                means.append(pred.mean)
                stds.append(pred.stddev)
        else:
            for head in self.dummy_heads:
                pred = head(features)
                means.append(pred.squeeze(-1))
                stds.append(torch.ones_like(pred).squeeze(-1) * 0.1)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)

def train_dkl(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test:  np.ndarray,
    Y_test:  np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    n_inducing: int = 128,
) -> Dict:
    n_outputs = Y_train.shape[1]
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Ytr = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)
    Yte = torch.tensor(Y_test,  dtype=torch.float32)

    model = DeepKernelLearning(
        input_dim=X_train.shape[1],
        n_outputs=n_outputs,
        feature_dim=32,
        n_inducing=n_inducing,
        X_init=Xtr,
    ).to(DEVICE)

    if not model.has_gpytorch:
        logger.info("[DKL] gpytorch not installed. Running dummy training.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            loss = 0.0
            for i in range(n_outputs):
                pred = model(Xtr, i)
                loss += torch.mean((pred - Ytr[:, i]) ** 2)
            loss.backward()
            optimizer.step()
        
        model.eval()
        mean_pred, std_pred = model.predict(Xte)
        mean_pred = mean_pred.cpu().numpy()
        std_pred  = std_pred.cpu().numpy()
    else:
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)

        mlls = [
            VariationalELBO(lik, gp, num_data=len(Xtr))
            for gp, lik in zip(model.gps, model.likelihoods)
        ]

        logger.info(f"[DKL] Training {n_epochs} epochs — {DEVICE}")

        for epoch in range(n_epochs):
            model.train()
            [gp.train() for gp in model.gps]
            [lik.train() for lik in model.likelihoods]

            optimizer.zero_grad()
            features = model.feature_net(Xtr)

            total_loss = 0.0
            for i, (gp, lik, mll) in enumerate(
                    zip(model.gps, model.likelihoods, mlls)):
                dist = gp(features)
                loss = -mll(dist, Ytr[:, i])
                total_loss = total_loss + loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        [gp.eval() for gp in model.gps]
        [lik.eval() for lik in model.likelihoods]

        mean_pred, std_pred = model.predict(Xte)
        mean_pred = mean_pred.cpu().numpy()
        std_pred  = std_pred.cpu().numpy()

    from sklearn.metrics import r2_score, mean_squared_error
    metrics = {}
    target_names = ["Vout_avg", "Efficiency", "IL_avg", "Vc_avg", "Iin_avg"]
    for i, name in enumerate(target_names[:n_outputs]):
        r2   = r2_score(Yte[:, i], mean_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Yte[:, i], mean_pred[:, i]))
        cal_err = np.mean(np.abs(Yte[:, i].detach().cpu().numpy() - mean_pred[:, i]) / (std_pred[:, i] + 1e-8))
        metrics[name] = {"R2": round(r2, 5), "RMSE": round(rmse, 5), "CalibrationErr": round(cal_err, 4)}

    out_dir = Path(__file__).resolve().parents[3] / "results" / "models" / "level1_dkl"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "dkl_model.pt")
    return metrics
