"""
Level 3 — Operator Learning Models
=====================================
Models: DeepONet, Fourier Neural Operator (FNO)
Target: Operating conditions -> Full Vout(t), IL(t) waveform trajectories
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.evaluation.metrics import compute_waveform_metrics, ModelTimer
from src.utils.visualization import (plot_training_history, plot_waveform_comparison,
                                      save_benchmark_json)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/level3_operator.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).resolve().parents[3]
WAVEFORM_DIR = BASE_DIR / "data" / "waveform_dataset"
MODELS_DIR   = BASE_DIR / "results" / "models" / "level3"
PLOTS_DIR    = BASE_DIR / "results" / "plots" / "level3"
BENCH_DIR    = BASE_DIR / "results" / "benchmarks"

for d in [MODELS_DIR, PLOTS_DIR, BENCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 400
BATCH_SIZE = 8
LR         = 1e-3
PATIENCE   = 50
N_SIGNALS  = 2


# ===========================================================================
# Data Loader
# ===========================================================================

def load_waveform_data():
    def _load(split):
        d = np.load(WAVEFORM_DIR / f"waveform_{split}.npz")
        X = torch.tensor(d['X'], dtype=torch.float32)
        Y = torch.tensor(d['Y'], dtype=torch.float32)  # (N, T, 2)
        T_arr = torch.tensor(d['T'], dtype=torch.float32)  # (N, T) normalized time
        return X, Y, T_arr

    X_train, Y_train, T_train = _load('train')
    X_val,   Y_val,   T_val   = _load('val')
    X_test,  Y_test,  T_test  = _load('test')

    T = Y_train.shape[1]
    logger.info(f"  Data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}, T={T}")
    return X_train, Y_train, T_train, X_val, Y_val, T_val, X_test, Y_test, T_test, T


# ===========================================================================
# DeepONet
# ===========================================================================

class BranchNet(nn.Module):
    """Branch network: encodes discrete operator input (operating conditions)."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, output_dim)


class TrunkNet(nn.Module):
    """Trunk network: encodes the query location (time)."""
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = None, output_dim: int = 128):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)  # (B, T, output_dim) or (T, output_dim)


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet).
    Learns: G(u)(y) = sum_k branch_k(u) * trunk_k(y) + bias
    u = operating conditions, y = time points
    """
    def __init__(self, branch_input: int = 6, trunk_input: int = 1,
                 p: int = 128, n_outputs: int = 2,
                 branch_hidden: List[int] = None, trunk_hidden: List[int] = None):
        super().__init__()
        branch_hidden = branch_hidden or [128, 256, 256, 128]
        trunk_hidden  = trunk_hidden  or [128, 256, 256, 128]

        # One branch + trunk pair per output signal
        self.branches = nn.ModuleList([
            BranchNet(branch_input, branch_hidden, p) for _ in range(n_outputs)
        ])
        self.trunks = nn.ModuleList([
            TrunkNet(trunk_input, trunk_hidden, p) for _ in range(n_outputs)
        ])
        self.biases = nn.Parameter(torch.zeros(n_outputs))
        self.n_outputs = n_outputs
        self.p = p

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_params)
        t : (B, T) — normalized time values

        Returns
        -------
        y : (B, T, n_outputs)
        """
        B, T = t.shape
        t_3d = t.unsqueeze(-1)  # (B, T, 1)

        outputs = []
        for k in range(self.n_outputs):
            b = self.branches[k](x)               # (B, p)
            tr = self.trunks[k](t_3d)             # (B, T, p)
            # Inner product over p dimension
            out_k = (b.unsqueeze(1) * tr).sum(-1) + self.biases[k]  # (B, T)
            outputs.append(out_k)

        return torch.stack(outputs, dim=-1)  # (B, T, n_outputs)


# ===========================================================================
# Fourier Neural Operator (FNO)
# ===========================================================================

class SpectralConv1d(nn.Module):
    """1D Fourier layer — key component of FNO."""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x, w):
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x):
        # x: (B, in_ch, T)
        B, _, T = x.shape
        x_ft = torch.fft.rfft(x)                           # (B, in_ch, T//2+1)
        out_ft = torch.zeros(B, self.out_channels, T // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=T)                # (B, out_ch, T)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spec_conv = SpectralConv1d(width, width, modes)
        self.bypass    = nn.Conv1d(width, width, 1)
        self.bn        = nn.BatchNorm1d(width)

    def forward(self, x):
        return torch.relu(self.bn(self.spec_conv(x) + self.bypass(x)))


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    Input:  (B, in_channels, T) where in_channels = n_params + 1 (time)
    Output: (B, n_outputs, T)
    """
    def __init__(self, in_channels: int = 7, out_channels: int = 2,
                 width: int = 64, modes: int = 32, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)

        self.fno_blocks = nn.Sequential(*[FNOBlock(width, modes) for _ in range(n_layers)])

        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(),
            nn.Conv1d(128, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_params) — scalar operating conditions
        t : (B, T)         — time grid

        Returns
        -------
        y : (B, T, n_outputs)
        """
        B, T = t.shape
        # Expand params across time dimension
        x_exp = x.unsqueeze(2).expand(-1, -1, T)  # (B, n_params, T)
        t_exp = t.unsqueeze(1)                      # (B, 1, T)
        inp   = torch.cat([x_exp, t_exp], dim=1)   # (B, n_params+1, T)

        h = self.lift(inp)              # (B, width, T)
        h = self.fno_blocks(h)         # (B, width, T)
        h = self.proj(h)               # (B, n_outputs, T)
        return h.permute(0, 2, 1)      # (B, T, n_outputs)


# ===========================================================================
# Training
# ===========================================================================

def train_operator_model(model: nn.Module, model_type: str,
                          X_train, Y_train, T_train,
                          X_val, Y_val, T_val) -> Tuple[List, List]:
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    Xtr = X_train.to(DEVICE); Ytr = Y_train.to(DEVICE); Ttr = T_train.to(DEVICE)
    Xvl = X_val.to(DEVICE);   Yvl = Y_val.to(DEVICE);   Tvl = T_val.to(DEVICE)

    train_ds = TensorDataset(Xtr, Ytr, Ttr)
    val_ds   = TensorDataset(Xvl, Yvl, Tvl)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    best_val = float('inf')
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0.0
        for xb, yb, tb in train_loader:
            optimizer.zero_grad()
            pred = model(xb, tb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(train_ds)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                vloss += criterion(model(xb, tb), yb).item() * xb.size(0)
        vloss /= len(val_ds)

        scheduler.step()
        train_losses.append(tloss)
        val_losses.append(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"    [{model_type}] Early stop at epoch {epoch}")
                break

        if epoch % 50 == 0:
            logger.info(f"    [{model_type}] Epoch {epoch:3d} | "
                        f"Train: {tloss:.6f} | Val: {vloss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_operator(model, model_type, X_test, Y_test, T_test) -> Tuple[Dict, np.ndarray]:
    model.eval().to(DEVICE)
    with torch.no_grad():
        with ModelTimer() as t_inf:
            Y_pred = model(X_test.to(DEVICE), T_test.to(DEVICE)).cpu().numpy()
    Y_true = Y_test.numpy()
    metrics = compute_waveform_metrics(Y_true, Y_pred, signal_names=['Vout', 'IL'])
    metrics['inference_time_ms'] = t_inf.elapsed * 1000
    return metrics, Y_pred


# ===========================================================================
# Level 3 Runner
# ===========================================================================

def run_level3() -> Dict:
    logger.info("=" * 70)
    logger.info("  LEVEL 3 — OPERATOR LEARNING MODELS")
    logger.info("=" * 70)

    (X_train, Y_train, T_train,
     X_val,   Y_val,   T_val,
     X_test,  Y_test,  T_test, T_len) = load_waveform_data()

    n_params = X_train.shape[1]
    benchmark = {}

    # ── DeepONet ──────────────────────────────────────────────────────────
    logger.info("\n  [DeepONet] Training...")
    deeponet = DeepONet(branch_input=n_params, trunk_input=1, p=128, n_outputs=N_SIGNALS)
    n_p = sum(p.numel() for p in deeponet.parameters())
    logger.info(f"    Parameters: {n_p:,}")

    with ModelTimer("DeepONet train") as t_train:
        tl, vl = train_operator_model(deeponet, "DeepONet",
                                       X_train, Y_train, T_train,
                                       X_val,   Y_val,   T_val)

    metrics_d, Ypred_d = evaluate_operator(deeponet, "DeepONet", X_test, Y_test, T_test)
    metrics_d['train_time_s'] = t_train.elapsed
    metrics_d['n_params'] = n_p
    benchmark['DeepONet'] = metrics_d
    torch.save({'model_state': deeponet.state_dict(),
                'config': {'model': 'DeepONet', 'n_params': n_params}},
               MODELS_DIR / "deeponet.pt")
    plot_training_history(tl, vl, "DeepONet", save_path=str(PLOTS_DIR / "deeponet_history.png"))

    # ── FNO ──────────────────────────────────────────────────────────────
    logger.info("\n  [FNO] Training Fourier Neural Operator...")
    fno = FNO1d(in_channels=n_params + 1, out_channels=N_SIGNALS, width=64, modes=32, n_layers=4)
    n_p_f = sum(p.numel() for p in fno.parameters())
    logger.info(f"    Parameters: {n_p_f:,}")

    with ModelTimer("FNO train") as t_train:
        tl_f, vl_f = train_operator_model(fno, "FNO",
                                           X_train, Y_train, T_train,
                                           X_val,   Y_val,   T_val)

    metrics_f, Ypred_f = evaluate_operator(fno, "FNO", X_test, Y_test, T_test)
    metrics_f['train_time_s'] = t_train.elapsed
    metrics_f['n_params'] = n_p_f
    benchmark['FNO'] = metrics_f
    torch.save({'model_state': fno.state_dict(),
                'config': {'model': 'FNO', 'n_params': n_params}},
               MODELS_DIR / "fno.pt")
    plot_training_history(tl_f, vl_f, "FNO", save_path=str(PLOTS_DIR / "fno_history.png"))

    # ── POD-DeepONet (Upgrade) ─────────────────────────────────────────────
    logger.info("\n  [POD-DeepONet] Training...")
    try:
        from src.models.pod_deeponet.pod_deeponet_model import PODDeepONet
        pod_onet = PODDeepONet(param_dim=n_params, pod_modes=32, n_signals=N_SIGNALS)
        pod_onet.fit_pod(Y_train.numpy())
        pod_onet = pod_onet.to(DEVICE)
        
        optimizer_pod = optim.AdamW(pod_onet.parameters(), lr=LR, weight_decay=1e-4)
        criterion_pod = nn.MSELoss()
        
        t0_pod = time.time()
        for epoch in range(1, 51):  # 50 epochs for fast feedback
            pod_onet.train()
            idx = torch.randperm(X_train.shape[0])
            for start in range(0, X_train.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, X_train.shape[0])
                xb = X_train[idx[start:end]].to(DEVICE)
                yb = Y_train[idx[start:end]].to(DEVICE)
                optimizer_pod.zero_grad()
                pred = pod_onet(xb)
                loss = criterion_pod(pred, yb)
                loss.backward()
                optimizer_pod.step()
        train_time_pod = time.time() - t0_pod
        
        pod_onet.eval()
        with torch.no_grad():
            t0_inf = time.time()
            Ypred_pod = pod_onet(X_test.to(DEVICE)).cpu().numpy()
            inf_time_pod = (time.time() - t0_inf) * 1000 / len(X_test)
            
        metrics_pod = compute_waveform_metrics(Y_test.numpy(), Ypred_pod, signal_names=['Vout', 'IL'])
        metrics_pod['train_time_s'] = train_time_pod
        metrics_pod['inference_time_ms'] = inf_time_pod
        metrics_pod['n_params'] = sum(p.numel() for p in pod_onet.parameters())
        benchmark['POD-DeepONet'] = metrics_pod
        torch.save({'model_state': pod_onet.state_dict(),
                    'config': {'model': 'POD-DeepONet', 'n_params': n_params}},
                   MODELS_DIR / "pod_deeponet.pt")
    except Exception as e:
        logger.error(f"POD-DeepONet training failed: {e}")

    # ── GINO (Upgrade) ─────────────────────────────────────────────────────
    logger.info("\n  [GINO] Training...")
    try:
        from src.models.gino.gino_model import GeometryInformedFNO
        gino_model = GeometryInformedFNO(param_dim=n_params, T=T_len, modes=32, width=64, n_layers=4)
        
        with ModelTimer("GINO train") as t_train:
            tl_g, vl_g = train_operator_model(gino_model, "GINO",
                                               X_train, Y_train, T_train,
                                               X_val,   Y_val,   T_val)
            
        metrics_g, Ypred_g = evaluate_operator(gino_model, "GINO", X_test, Y_test, T_test)
        metrics_g['train_time_s'] = t_train.elapsed
        metrics_g['n_params'] = sum(p.numel() for p in gino_model.parameters())
        benchmark['GINO'] = metrics_g
        torch.save({'model_state': gino_model.state_dict(),
                    'config': {'model': 'GINO', 'n_params': n_params}},
                   MODELS_DIR / "gino.pt")
    except Exception as e:
        logger.error(f"GINO training failed: {e}")

    # Waveform comparisons
    T_axis = np.linspace(0, 1, T_len)
    for name, Ypred in [("DeepONet", Ypred_d), ("FNO", Ypred_f)]:
        for s_idx, s_name in enumerate(['Vout', 'IL']):
            plot_waveform_comparison(
                T_axis,
                Y_test[:1, :, s_idx].numpy().T,
                Ypred[:1, :, s_idx].T,
                labels=[s_name],
                title=f"{name} — {s_name} Waveform",
                save_path=str(PLOTS_DIR / f"{name.lower()}_{s_name.lower()}_waveform.png")
            )

    # Summary
    save_benchmark_json(benchmark, str(BENCH_DIR / "level3_benchmark.json"))
    logger.info("\n" + "=" * 70)
    logger.info("  LEVEL 3 RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<12} {'R²(Vout)':>10} {'R²(IL)':>10} {'Train(s)':>10}")
    logger.info("  " + "-" * 45)
    for name, m in benchmark.items():
        logger.info(f"  {name:<12} {m.get('Vout_R2', float('nan')):>10.4f} "
                    f"{m.get('IL_R2', float('nan')):>10.4f} "
                    f"{m.get('train_time_s', float('nan')):>10.2f}")

    logger.info("\n[DONE] LEVEL 3 COMPLETE")
    return benchmark


if __name__ == "__main__":
    run_level3()
