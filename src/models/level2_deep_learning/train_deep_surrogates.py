"""
Level 2 — Deep Learning Surrogates
=====================================
Models: 1D CNN, TCN, Transformer Encoder, Autoencoder + Regression Head
Target: Waveform prediction (Vout(t), IL(t)) from scalar operating conditions
"""

import sys
import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.evaluation.metrics import compute_waveform_metrics, ModelTimer
from src.utils.visualization import (plot_training_history, plot_waveform_comparison,
                                      plot_benchmark_comparison, save_benchmark_json)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/level2_deep_learning.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).resolve().parents[3]
WAVEFORM_DIR = BASE_DIR / "data" / "waveform_dataset"
MODELS_DIR   = BASE_DIR / "results" / "models" / "level2"
PLOTS_DIR    = BASE_DIR / "results" / "plots" / "level2"
BENCH_DIR    = BASE_DIR / "results" / "benchmarks"

for d in [MODELS_DIR, PLOTS_DIR, BENCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"  Device: {DEVICE}")

EPOCHS     = 300
BATCH_SIZE = 8
LR         = 1e-3
PATIENCE   = 40
N_SIGNALS  = 2     # Vout, IL


# ===========================================================================
# Dataset Loader
# ===========================================================================

def load_waveform_data():
    """Load normalized waveform datasets."""
    def _load(split):
        d = np.load(WAVEFORM_DIR / f"waveform_{split}.npz")
        X = torch.tensor(d['X'], dtype=torch.float32)
        Y = torch.tensor(d['Y'], dtype=torch.float32)   # (N, T, 2)
        return X, Y

    X_train, Y_train = _load('train')
    X_val,   Y_val   = _load('val')
    X_test,  Y_test  = _load('test')

    T = Y_train.shape[1]
    logger.info(f"  Waveform data — Train: {X_train.shape}, T={T}, Signals={N_SIGNALS}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, T


# ===========================================================================
# Model Architectures
# ===========================================================================

class CNN1D(nn.Module):
    """1D CNN: params -> waveform via transposed convolutions."""

    def __init__(self, n_params: int = 6, T: int = 512, n_signals: int = 2):
        super().__init__()
        self.T = T
        self.n_signals = n_signals
        self.stem = nn.Sequential(
            nn.Linear(n_params, 256), nn.ReLU(),
            nn.Linear(256, 128 * 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, n_signals, kernel_size=4, stride=2, padding=1),
        )
        self.adapt = nn.AdaptiveAvgPool1d(T)

    def forward(self, x):
        h = self.stem(x).view(x.size(0), 128, 8)
        h = self.decoder(h)
        h = self.adapt(h)  # (B, 2, T)
        return h.permute(0, 2, 1)  # (B, T, 2)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrize.remove_parametrizations if False else nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.chomp = lambda x: x[:, :, :-(padding)] if padding else x
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.chomp(self.conv1(x))))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.chomp(self.conv2(out))))
        out = self.dropout(out)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network: params -> waveform."""

    def __init__(self, n_params: int = 6, T: int = 512, n_signals: int = 2,
                 channels: List[int] = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        channels = channels or [64, 64, 128, 128]
        self.T = T
        self.n_signals = n_signals

        # Param encoder -> initial sequence
        self.param_enc = nn.Sequential(
            nn.Linear(n_params, 128), nn.ReLU(),
            nn.Linear(128, channels[0] * T // 8)
        )
        self.T_init = T // 8

        # TCN blocks
        tcn_layers = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(channels[-1], 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_signals, 4, stride=2, padding=1),
        )
        self.adapt = nn.AdaptiveAvgPool1d(T)

    def forward(self, x):
        B = x.size(0)
        h = self.param_enc(x).view(B, -1, self.T_init)
        h = self.tcn(h)
        h = self.upsample(h)
        h = self.adapt(h)
        return h.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerSurrogate(nn.Module):
    """Transformer Encoder: params -> waveform via cross-attention."""

    def __init__(self, n_params: int = 6, T: int = 512, n_signals: int = 2,
                 d_model: int = 128, nhead: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.d_model = d_model

        # Param embedding
        self.param_embed = nn.Sequential(
            nn.Linear(n_params, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        # Time query embedding
        self.time_embed = nn.Embedding(T, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=T + 1)

        # Transformer encoder (self-attention over time queries)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Conditioning: add param context to each time step
        self.cond_proj = nn.Linear(d_model, d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_signals)
        )

    def forward(self, x):
        B = x.size(0)
        # Param context: (B, d_model)
        ctx = self.param_embed(x)

        # Time tokens: (B, T, d_model)
        t_idx = torch.arange(self.T, device=x.device).unsqueeze(0).expand(B, -1)
        q = self.time_embed(t_idx)
        q = self.pos_enc(q)

        # Inject parameter context
        q = q + self.cond_proj(ctx).unsqueeze(1)

        # Transformer
        out = self.transformer(q)  # (B, T, d_model)
        return self.head(out)       # (B, T, n_signals)


class AutoencoderRegressor(nn.Module):
    """Autoencoder + Regression Head."""

    def __init__(self, n_params: int = 6, T: int = 512, n_signals: int = 2, latent_dim: int = 32):
        super().__init__()
        self.T = T
        self.n_signals = n_signals

        # Encoder (from waveform to latent) — used during pre-training
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * n_signals, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder (from latent to waveform)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, T * n_signals)
        )

        # Regression head (from params to latent)
        self.regressor = nn.Sequential(
            nn.Linear(n_params, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def encode(self, y):  # y: (B, T, S)
        return self.encoder(y)

    def decode(self, z):  # z: (B, latent)
        out = self.decoder(z)
        return out.view(out.size(0), self.T, self.n_signals)

    def forward(self, x):  # x: (B, n_params)
        z = self.regressor(x)
        return self.decode(z)

    def autoencoder_forward(self, y):
        z = self.encode(y)
        return self.decode(z), z


# ===========================================================================
# Training Loop
# ===========================================================================

def train_model(model: nn.Module, X_train, Y_train, X_val, Y_val,
                model_name: str, lr: float = LR,
                epochs: int = EPOCHS, patience: int = PATIENCE) -> Tuple[List, List]:

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    train_ds = TensorDataset(X_train.to(DEVICE), Y_train.to(DEVICE))
    val_ds   = TensorDataset(X_val.to(DEVICE), Y_val.to(DEVICE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val = float('inf')
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"    Early stop at epoch {epoch}")
                break

        if epoch % 50 == 0:
            logger.info(f"    {model_name} | Epoch {epoch:3d} | "
                        f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    return train_losses, val_losses


def evaluate_model(model: nn.Module, X_test, Y_test) -> Tuple[Dict, np.ndarray]:
    model.eval().to(DEVICE)
    with torch.no_grad():
        with ModelTimer() as t_inf:
            Y_pred = model(X_test.to(DEVICE)).cpu().numpy()
    Y_true = Y_test.numpy()
    metrics = compute_waveform_metrics(Y_true, Y_pred, signal_names=['Vout', 'IL'])
    metrics['inference_time_ms'] = t_inf.elapsed * 1000
    return metrics, Y_pred


# ===========================================================================
# Level 2 Runner
# ===========================================================================

def run_level2() -> Dict:
    logger.info("=" * 70)
    logger.info("  LEVEL 2 — DEEP LEARNING SURROGATES")
    logger.info("=" * 70)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, T = load_waveform_data()
    n_params = X_train.shape[1]
    benchmark = {}

    model_configs = [
        ("CNN1D",       CNN1D(n_params=n_params, T=T, n_signals=N_SIGNALS)),
        ("TCN",         TCN(n_params=n_params, T=T, n_signals=N_SIGNALS)),
        ("Transformer", TransformerSurrogate(n_params=n_params, T=T, n_signals=N_SIGNALS)),
        ("Autoencoder", AutoencoderRegressor(n_params=n_params, T=T, n_signals=N_SIGNALS)),
    ]

    for name, model in model_configs:
        ckpt_path = MODELS_DIR / f"{name.lower()}.pt"

        # ── Skip if checkpoint already exists ──────────────────────────
        if ckpt_path.exists():
            logger.info(f"\n  [{name}] Checkpoint found — loading saved model (skipping retrain)")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            metrics, Y_pred = evaluate_model(model, X_test, Y_test)
            metrics['train_time_s'] = 0.0   # already trained
            metrics['n_params'] = sum(p.numel() for p in model.parameters())
            benchmark[name] = metrics
            r2 = metrics.get('overall_R2', float('nan'))
            logger.info(f"    {name} | overall R2={r2:.4f} | [loaded from checkpoint]")
            continue
        # ───────────────────────────────────────────────────────────────

        logger.info(f"\n  [{name}] Training...")
        n_params_model = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params_model:,}")

        with ModelTimer(f"{name} train") as t_train:
            train_losses, val_losses = train_model(
                model, X_train, Y_train, X_val, Y_val, name
            )

        metrics, Y_pred = evaluate_model(model, X_test, Y_test)
        metrics['train_time_s'] = t_train.elapsed
        metrics['n_params'] = n_params_model
        benchmark[name] = metrics

        r2 = metrics.get('overall_R2', float('nan'))
        logger.info(f"    {name} | overall R2={r2:.4f} | "
                    f"Train: {t_train.elapsed:.1f}s | "
                    f"Infer: {metrics['inference_time_ms']:.1f}ms")

        # Save model
        torch.save({'model_state': model.state_dict(),
                    'config': {'name': name, 'n_params': n_params,
                               'T': T, 'n_signals': N_SIGNALS}},
                   ckpt_path)

        # Plots
        plot_training_history(train_losses, val_losses, model_name=name,
                              save_path=str(PLOTS_DIR / f"{name.lower()}_history.png"))

        # Waveform comparison (first test sample)
        T_axis = np.linspace(0, 1, T)
        for s_idx, s_name in enumerate(['Vout', 'IL']):
            plot_waveform_comparison(
                T_axis, Y_test[:1, :, s_idx].numpy().T, Y_pred[:1, :, s_idx].T,
                labels=[s_name],
                title=f"{name} — {s_name} Waveform (Sample 0)",
                save_path=str(PLOTS_DIR / f"{name.lower()}_{s_name.lower()}_waveform.png")
            )

    # Save benchmark
    save_benchmark_json(benchmark, str(BENCH_DIR / "level2_benchmark.json"))

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("  LEVEL 2 RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<15} {'R2(Vout)':>10} {'R2(IL)':>10} {'Train(s)':>10}")
    logger.info("  " + "-" * 55)
    for name, m in benchmark.items():
        r2_v = m.get('Vout_R2', float('nan'))
        r2_i = m.get('IL_R2', float('nan'))
        ttr  = m.get('train_time_s', float('nan'))
        tag  = "  [loaded]" if ttr == 0.0 else ""
        logger.info(f"  {name:<15} {r2_v:>10.4f} {r2_i:>10.4f} {ttr:>10.2f}{tag}")

    logger.info("\n[DONE] LEVEL 2 COMPLETE")
    return benchmark


if __name__ == "__main__":
    run_level2()
