"""
level1_upgrade_dkl.py
=====================
REPLACES: GPR (Matern kernel), SVR
WITH:      Deep Kernel Learning (DKL) — GPyTorch exact GP on top of a
           learned deep feature extractor, giving O(n) sparse inducing-point
           approximation (SVGP) plus calibrated uncertainty.

WHY HIGHER-LEVEL:
  - Standard GPR is O(n³) — unusable beyond ~2000 samples.
  - SVR learns no uncertainty and has fixed kernel structure.
  - DKL: neural net learns a non-stationary, task-specific kernel; SVGP
    scales to 100k+ samples with inducing points; calibrated posterior
    gives confidence intervals, not just point estimates.

Targets: scalar outputs [Vout_avg, Efficiency, IL_avg, Vc_avg, Iin_avg]
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
    print("[WARN] gpytorch not installed. Run: pip install gpytorch")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[1]
SCALAR_DIR = BASE_DIR / "data" / "scalar_dataset"
OUT_DIR    = BASE_DIR / "results" / "models" / "level1_dkl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Deep Feature Extractor ─────────────────────────────────────────────────
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


# ─── Sparse Variational GP (SVGP) ────────────────────────────────────────────
class SVGPModel(ApproximateGP):
    """
    Sparse Variational GP with inducing points in the learned feature space.
    Scales to large datasets; provides calibrated uncertainty.
    """
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy)

        # Composite kernel: Matern(5/2) × RBF + Linear
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5) * RBFKernel() + LinearKernel()
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )


# ─── DKL Model (Feature Net + GP per target) ─────────────────────────────────
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

        # Initialise inducing points from real data distribution
        if X_init is not None:
            # Project data to feature space for sensible initialization
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

    def forward(
        self, x: torch.Tensor, target_idx: int
    ) -> MultivariateNormal:
        features = self.feature_net(x)
        return self.gps[target_idx](features)

    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) for all outputs stacked as (B, n_outputs)."""
        means, stds = [], []
        features = self.feature_net(x)
        for gp, lik in zip(self.gps, self.likelihoods):
            gp.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = lik(gp(features))
            means.append(pred.mean)
            stds.append(pred.stddev)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_dkl(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test:  np.ndarray,
    Y_test:  np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    n_inducing: int = 128,
) -> Dict:
    if not HAS_GPYTORCH:
        return {"error": "gpytorch not installed"}

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

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    # Train each GP head jointly
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

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                        f"ELBO Loss: {total_loss.item():.4f}")

    # Evaluate
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
        cal_err = np.mean(np.abs(Yte[:, i].numpy() - mean_pred[:, i]) /
                          (std_pred[:, i] + 1e-8))
        metrics[name] = {"R2": round(r2, 5), "RMSE": round(rmse, 5),
                         "CalibrationErr": round(cal_err, 4)}
        logger.info(f"  {name:<14}: R²={r2:.4f}  RMSE={rmse:.4f}  "
                    f"CalibErr={cal_err:.4f}")

    torch.save(model.state_dict(), OUT_DIR / "dkl_model.pt")
    json.dump(metrics, open(OUT_DIR / "dkl_metrics.json", "w"), indent=2)
    return metrics


if __name__ == "__main__":
    # Demo with random data (replace with real LTspice data)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 7)).astype(np.float32)
    Y = rng.standard_normal((500, 5)).astype(np.float32)
    train_dkl(X[:400], Y[:400], X[400:], Y[400:], n_epochs=100)
