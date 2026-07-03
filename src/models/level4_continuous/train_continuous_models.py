"""
Level 4 — Continuous-Time Dynamic Models
==========================================
Models: Neural ODE, Latent Neural ODE, ODE-RNN
Learns: dx/dt = f(x, t, θ) — continuous state dynamics of the boost converter

State vector: x = [IL, Vout]
Parameters:   u = [Vin, D, Fs, L, C, Rload]
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
from torch.utils.data import Dataset, DataLoader

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
        logging.FileHandler("logs/level4_continuous.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).resolve().parents[3]
DYNAMIC_DIR  = BASE_DIR / "data" / "dynamic_dataset"
WAVEFORM_DIR = BASE_DIR / "data" / "waveform_dataset"
MODELS_DIR   = BASE_DIR / "results" / "models" / "level4"
PLOTS_DIR    = BASE_DIR / "results" / "plots" / "level4"
BENCH_DIR    = BASE_DIR / "results" / "benchmarks"

for d in [MODELS_DIR, PLOTS_DIR, BENCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"  Device: {DEVICE}")

STATE_DIM  = 2    # [IL, Vout]
PARAM_DIM  = 6    # [Vin, D, Fs, L, C, Rload]
EPOCHS     = 200
BATCH_SIZE = 4
LR         = 5e-4
PATIENCE   = 30
N_TIME_EVAL = 64  # Reduced time points for ODE evaluation speed


# ===========================================================================
# Check for torchdiffeq
# ===========================================================================

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
    logger.info("  torchdiffeq available [OK]")
except ImportError:
    HAS_TORCHDIFFEQ = False
    logger.warning("  torchdiffeq not found — using RK4 manual integration")


def rk4_step(f, t, x, dt):
    """Manual RK4 step for fallback."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(f, x0, t_span):
    """Integrate ODE using RK4."""
    x = x0
    traj = [x0]
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        x = rk4_step(f, t_span[i], x, dt)
        traj.append(x)
    return torch.stack(traj, dim=1)  # (B, T, state_dim)


# ===========================================================================
# Dataset
# ===========================================================================

class DynamicDataset(Dataset):
    """
    Loads per-trajectory data for Neural ODE training.
    Each sample = (params, initial_state, time_points, state_trajectory)
    """
    def __init__(self, split: str, n_time: int = N_TIME_EVAL):
        self.data = np.load(DYNAMIC_DIR / f"dynamic_{split}.npz", allow_pickle=True)
        self.n_time = n_time
        # Count samples
        self.n = sum(1 for k in self.data.files if k.startswith("params_"))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        params = torch.tensor(self.data[f"params_{idx}"], dtype=torch.float32)
        time   = torch.tensor(self.data[f"time_{idx}"],   dtype=torch.float32)
        state  = torch.tensor(self.data[f"state_{idx}"],  dtype=torch.float32)

        # Subsample to n_time points uniformly
        n_raw = len(time)
        indices = np.linspace(0, n_raw - 1, self.n_time).astype(int)
        time_sub  = time[indices]
        state_sub = state[indices]

        # Normalize time to [0, 1]
        t_min, t_max = time_sub[0], time_sub[-1]
        time_norm = (time_sub - t_min) / (t_max - t_min + 1e-12)

        return params, time_norm, state_sub


# ===========================================================================
# Neural ODE
# ===========================================================================

class ODEFunc(nn.Module):
    """
    Neural ODE dynamics: dx/dt = f(x, t, u)
    where u = operating conditions (injected as context).
    """
    def __init__(self, state_dim: int = STATE_DIM, param_dim: int = PARAM_DIM,
                 hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]

        # Context encoder
        self.param_enc = nn.Sequential(
            nn.Linear(param_dim, 128), nn.Tanh(),
            nn.Linear(128, 64)
        )

        # Main ODE network: input = [x, t, context]
        in_dim = state_dim + 1 + 64
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Softplus()]
            in_dim = h
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

        self._context = None

    def set_context(self, params: torch.Tensor):
        """Set operating condition context before integration."""
        self._context = self.param_enc(params)  # (B, 64)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : scalar tensor
        x : (B, state_dim)

        Returns
        -------
        dxdt : (B, state_dim)
        """
        B = x.shape[0]
        t_exp = t.expand(B, 1) if t.dim() == 0 else t.unsqueeze(-1)
        ctx = self._context if self._context is not None else torch.zeros(B, 64, device=x.device)
        inp = torch.cat([x, t_exp, ctx], dim=-1)
        return self.net(inp)


class NeuralODE(nn.Module):
    """Neural ODE surrogate for boost converter dynamics."""

    def __init__(self, state_dim: int = STATE_DIM, param_dim: int = PARAM_DIM,
                 hidden_dims: List[int] = None):
        super().__init__()
        self.ode_func = ODEFunc(state_dim=state_dim, param_dim=param_dim,
                                hidden_dims=hidden_dims)
        self.state_dim = state_dim

    def forward(self, params: torch.Tensor, x0: torch.Tensor,
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : (B, param_dim)
        x0     : (B, state_dim)
        t_span : (T,) time points

        Returns
        -------
        traj : (B, T, state_dim)
        """
        self.ode_func.set_context(params)

        if HAS_TORCHDIFFEQ:
            traj = odeint(self.ode_func, x0, t_span,
                          method='rk4', options={'step_size': 0.05})
            return traj.permute(1, 0, 2)  # (B, T, state_dim)
        else:
            return integrate_rk4(self.ode_func, x0, t_span)


# ===========================================================================
# Latent Neural ODE
# ===========================================================================

class LatentNeuralODE(nn.Module):
    """
    Latent Neural ODE:
    Encoder: waveform initial segment -> latent z0
    ODE:     dz/dt = f(z, t, params)
    Decoder: z(t) -> state(t)
    """
    def __init__(self, state_dim: int = STATE_DIM, param_dim: int = PARAM_DIM,
                 latent_dim: int = 16, obs_steps: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_steps  = obs_steps

        # Encoder: obs_steps states -> latent z0
        self.encoder = nn.Sequential(
            nn.Linear(obs_steps * state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # mean + logvar
        )

        # ODE in latent space
        self.ode_func_latent = nn.Sequential(
            nn.Linear(latent_dim + param_dim + 1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: latent -> state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, state_dim)
        )

        # Param context for ODE
        self._params = None

    def encode(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_obs: (B, obs_steps, state_dim)"""
        flat = x_obs.flatten(1)
        h = self.encoder(flat)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def _ode_step(self, t, z):
        B = z.shape[0]
        t_exp = t.expand(B, 1) if t.dim() == 0 else t.unsqueeze(-1)
        p = self._params if self._params is not None else torch.zeros(B, PARAM_DIM, device=z.device)
        inp = torch.cat([z, p, t_exp], dim=-1)
        return self.ode_func_latent(inp)

    def forward(self, params: torch.Tensor, x_obs: torch.Tensor,
                t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (trajectory, mean, logvar)
        """
        self._params = params
        mean, logvar = self.encode(x_obs)
        z0 = self.reparameterize(mean, logvar)

        if HAS_TORCHDIFFEQ:
            z_traj = odeint(self._ode_step, z0, t_span,
                            method='rk4', options={'step_size': 0.05})
            z_traj = z_traj.permute(1, 0, 2)  # (B, T, latent)
        else:
            z_traj = integrate_rk4(self._ode_step, z0, t_span)

        traj = self.decoder(z_traj)  # (B, T, state_dim)
        return traj, mean, logvar


# ===========================================================================
# ODE-RNN
# ===========================================================================

class ODERNN(nn.Module):
    """
    ODE-RNN: integrates ODE between observations, updates with RNN at each point.
    Simplified version for batch training.
    """
    def __init__(self, state_dim: int = STATE_DIM, param_dim: int = PARAM_DIM,
                 hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size

        # Param encoder
        self.param_enc = nn.Sequential(
            nn.Linear(param_dim, 64), nn.ReLU(), nn.Linear(64, hidden_size)
        )

        # GRU for state updates
        self.rnn = nn.GRU(state_dim + 1, hidden_size, n_layers,
                          batch_first=True, dropout=0.1 if n_layers > 1 else 0)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, state_dim)

    def forward(self, params: torch.Tensor, t_span: torch.Tensor,
                x0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : (B, param_dim)
        t_span : (B, T)  normalized time
        x0     : (B, state_dim)

        Returns
        -------
        traj : (B, T, state_dim)
        """
        B, T = t_span.shape
        h0 = self.param_enc(params).unsqueeze(0).expand(self.rnn.num_layers, -1, -1)
        h0 = h0.contiguous()

        # Concatenate x0 with t=0 -> input at first step
        t_exp = t_span.unsqueeze(-1)  # (B, T, 1)
        # Use x0 repeated as input (no teacher forcing)
        x_rep = x0.unsqueeze(1).expand(-1, T, -1)  # (B, T, state_dim)
        inp = torch.cat([x_rep, t_exp], dim=-1)   # (B, T, state_dim+1)

        out, _ = self.rnn(inp, h0)
        return self.out_proj(out)  # (B, T, state_dim)


# ===========================================================================
# Training
# ===========================================================================

def train_neural_ode(model_name: str, model: nn.Module) -> Tuple[List, List, Dict]:
    logger.info(f"\n  [{model_name}] Training on DEVICE={DEVICE}...")
    n_p = sum(p.numel() for p in model.parameters())
    logger.info(f"    Parameters: {n_p:,}")

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    try:
        train_ds = DynamicDataset('train')
        val_ds   = DynamicDataset('val')
        test_ds  = DynamicDataset('test')
    except Exception as e:
        logger.error(f"    Dataset load error: {e}")
        return [], [], {}

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1, collate_fn=_collate)

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    no_improve = 0
    criterion = nn.MSELoss()

    with ModelTimer(f"{model_name} train") as t_train:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            tloss = 0.0
            for params, t_span, state in train_loader:
                params = params.to(DEVICE)
                t_span = t_span.to(DEVICE)
                state  = state.to(DEVICE)
                x0 = state[:, 0, :]
                t0 = t_span[0]

                optimizer.zero_grad()
                try:
                    if isinstance(model, NeuralODE):
                        pred = model(params, x0, t0)
                    elif isinstance(model, LatentNeuralODE):
                        obs = state[:, :model.obs_steps, :]
                        pred, mu, lv = model(params, obs, t0)
                        kl_loss = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    elif isinstance(model, ODERNN):
                        pred = model(params, t_span, x0)
                except Exception as e:
                    continue

                recon_loss = criterion(pred, state)
                loss = recon_loss
                if isinstance(model, LatentNeuralODE):
                    loss = loss + 0.001 * kl_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tloss += loss.item()

            tloss /= max(len(train_loader), 1)
            train_losses.append(tloss)

            # Validation
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for params, t_span, state in val_loader:
                    params = params.to(DEVICE); t_span = t_span.to(DEVICE); state = state.to(DEVICE)
                    x0 = state[:, 0, :]
                    t0 = t_span[0]
                    try:
                        if isinstance(model, NeuralODE):
                            pred = model(params, x0, t0)
                        elif isinstance(model, LatentNeuralODE):
                            obs = state[:, :model.obs_steps, :]
                            pred, _, _ = model(params, obs, t0)
                        elif isinstance(model, ODERNN):
                            pred = model(params, t_span, x0)
                        vloss += criterion(pred, state).item()
                    except:
                        pass
            vloss /= max(len(val_loader), 1)
            val_losses.append(vloss)
            scheduler.step(vloss)

            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    logger.info(f"    [{model_name}] Early stop at epoch {epoch}")
                    break

            if epoch % 50 == 0:
                logger.info(f"    [{model_name}] Epoch {epoch:3d} | "
                            f"Train: {tloss:.6f} | Val: {vloss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluation on test set
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for params, t_span, state in test_loader:
            params = params.to(DEVICE); t_span = t_span.to(DEVICE); state = state.to(DEVICE)
            x0 = state[:, 0, :]
            t0 = t_span[0]
            try:
                if isinstance(model, NeuralODE):
                    pred = model(params, x0, t0)
                elif isinstance(model, LatentNeuralODE):
                    obs = state[:, :model.obs_steps, :]
                    pred, _, _ = model(params, obs, t0)
                elif isinstance(model, ODERNN):
                    pred = model(params, t_span, x0)
                all_true.append(state.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
            except:
                pass

    metrics = {}
    if all_true:
        yt = np.concatenate(all_true, axis=0)
        yp = np.concatenate(all_pred, axis=0)
        metrics = compute_waveform_metrics(yt, yp, signal_names=['IL', 'Vout'])

    metrics['train_time_s'] = t_train.elapsed
    metrics['n_params'] = n_p

    torch.save({'model_state': model.state_dict()}, MODELS_DIR / f"{model_name.lower().replace('-', '_')}.pt")
    logger.info(f"    [{model_name}] R²(IL)={metrics.get('IL_R2', float('nan')):.4f}, "
                f"R²(Vout)={metrics.get('Vout_R2', float('nan')):.4f}, "
                f"Train: {t_train.elapsed:.1f}s")

    return train_losses, val_losses, metrics


def _collate(batch):
    """Collate function for variable-length trajectories (pads to same length)."""
    params_list = [item[0] for item in batch]
    t_list      = [item[1] for item in batch]
    state_list  = [item[2] for item in batch]

    # Use minimum length for batch
    min_len = min(t.shape[0] for t in t_list)

    params_t = torch.stack(params_list)
    t_t      = torch.stack([t[:min_len] for t in t_list])
    state_t  = torch.stack([s[:min_len] for s in state_list])

    return params_t, t_t, state_t


# ===========================================================================
# Level 4 Runner
# ===========================================================================

def run_level4() -> Dict:
    logger.info("=" * 70)
    logger.info("  LEVEL 4 — CONTINUOUS-TIME DYNAMIC MODELS")
    logger.info("=" * 70)

    benchmark = {}

    models_to_train = [
        ("NeuralODE",      NeuralODE(state_dim=STATE_DIM, param_dim=PARAM_DIM)),
        ("LatentNeuralODE", LatentNeuralODE(state_dim=STATE_DIM, param_dim=PARAM_DIM,
                                            latent_dim=16, obs_steps=4)),
        ("ODE-RNN",         ODERNN(state_dim=STATE_DIM, param_dim=PARAM_DIM,
                                   hidden_size=128, n_layers=2)),
    ]

    for model_name, model in models_to_train:
        try:
            tl, vl, metrics = train_neural_ode(model_name, model)
            benchmark[model_name] = metrics
            if tl and vl:
                plot_training_history(tl, vl, model_name,
                                      save_path=str(PLOTS_DIR / f"{model_name.lower()}_history.png"))
        except Exception as e:
            logger.error(f"  [{model_name}] Failed: {e}")
            benchmark[model_name] = {'error': str(e)}

    # Summary
    save_benchmark_json(benchmark, str(BENCH_DIR / "level4_benchmark.json"))

    logger.info("\n" + "=" * 70)
    logger.info("  LEVEL 4 RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<18} {'R²(IL)':>10} {'R²(Vout)':>10} {'Train(s)':>10}")
    logger.info("  " + "-" * 55)
    for name, m in benchmark.items():
        if 'error' not in m:
            logger.info(f"  {name:<18} {m.get('IL_R2', float('nan')):>10.4f} "
                        f"{m.get('Vout_R2', float('nan')):>10.4f} "
                        f"{m.get('train_time_s', float('nan')):>10.2f}")
        else:
            logger.info(f"  {name:<18} ERROR: {m['error']}")

    logger.info("\n[DONE] LEVEL 4 COMPLETE")
    return benchmark


if __name__ == "__main__":
    run_level4()
