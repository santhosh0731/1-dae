"""
Visualization Utilities
=======================
Publication-quality plots for surrogate model evaluation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
import json


STYLE = {
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',
    'axes.labelcolor': '#C9D1D9',
    'axes.titlecolor': '#C9D1D9',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'text.color': '#C9D1D9',
    'grid.color': '#21262D',
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
}

PALETTE = ['#58A6FF', '#3FB950', '#F78166', '#D2A8FF', '#79C0FF',
           '#56D364', '#FFA657', '#FF7B72', '#A5D6FF', '#7EE787']


def apply_style():
    plt.rcParams.update(STYLE)


def plot_waveform_comparison(
    time: np.ndarray,
    true_waveform: np.ndarray,
    pred_waveform: np.ndarray,
    labels: List[str],
    title: str = "Waveform Comparison",
    save_path: Optional[str] = None
):
    """Plot predicted vs. true waveforms (multi-signal)."""
    apply_style()
    n_signals = true_waveform.shape[1] if true_waveform.ndim > 1 else 1
    if true_waveform.ndim == 1:
        true_waveform = true_waveform[:, None]
        pred_waveform = pred_waveform[:, None]

    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3 * n_signals),
                              facecolor=STYLE['figure.facecolor'])
    if n_signals == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_facecolor(STYLE['axes.facecolor'])
        ax.plot(time * 1e3, true_waveform[:, i], color=PALETTE[0],
                label='LTspice (True)', linewidth=2, alpha=0.9)
        ax.plot(time * 1e3, pred_waveform[:, i], color=PALETTE[2],
                label='Predicted', linewidth=1.5, linestyle='--', alpha=0.9)
        ax.set_ylabel(label, color=STYLE['axes.labelcolor'])
        ax.legend(loc='upper right', framealpha=0.4)
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE['axes.edgecolor'])

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=14, color='#E6EDF3', fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['figure.facecolor'])
        print(f"  [Plot] Saved: {save_path}")
    plt.close()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """Plot training and validation loss curves."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=STYLE['figure.facecolor'])
    ax.set_facecolor(STYLE['axes.facecolor'])

    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, color=PALETTE[0], label='Train Loss', linewidth=2)
    ax.semilogy(epochs, val_losses, color=PALETTE[2], label='Val Loss', linewidth=2,
                linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"{model_name} — Training History", fontweight='bold')
    ax.legend(framealpha=0.4)
    ax.grid(True, which='both', alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['axes.edgecolor'])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['figure.facecolor'])
        print(f"  [Plot] Saved: {save_path}")
    plt.close()


def plot_benchmark_comparison(
    benchmark_data: Dict[str, Dict],
    metric: str = "R2",
    title: str = "Model Benchmark",
    save_path: Optional[str] = None
):
    """Horizontal bar chart comparing all models on a given metric."""
    apply_style()
    models = list(benchmark_data.keys())
    values = []
    for m in models:
        v = benchmark_data[m].get(metric, float('nan'))
        values.append(v if v is not None else float('nan'))

    # Sort
    sorted_pairs = sorted(zip(values, models), key=lambda x: x[0] if not np.isnan(x[0]) else -1)
    values_s, models_s = zip(*sorted_pairs)

    colors = [PALETTE[3] if v == max(values_s) else PALETTE[0] for v in values_s]

    fig, ax = plt.subplots(figsize=(10, max(5, len(models) * 0.6)),
                           facecolor=STYLE['figure.facecolor'])
    ax.set_facecolor(STYLE['axes.facecolor'])
    bars = ax.barh(models_s, values_s, color=colors, alpha=0.85, height=0.6)

    for bar, val in zip(bars, values_s):
        if not np.isnan(val):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', color='#E6EDF3', fontsize=9)

    ax.set_xlabel(metric)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['axes.edgecolor'])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['figure.facecolor'])
        print(f"  [Plot] Saved: {save_path}")
    plt.close()


def plot_scatter_true_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Output",
    r2: float = None,
    save_path: Optional[str] = None
):
    """Scatter plot of true vs. predicted values."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=STYLE['figure.facecolor'])
    ax.set_facecolor(STYLE['axes.facecolor'])

    ax.scatter(y_true, y_pred, color=PALETTE[0], alpha=0.7, s=40, edgecolors='none')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=PALETTE[2],
            linewidth=1.5, linestyle='--', label='Perfect fit')

    title_str = f"{target_name}: True vs Predicted"
    if r2 is not None:
        title_str += f"  |  R² = {r2:.4f}"
    ax.set_title(title_str, fontweight='bold')
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.legend(framealpha=0.4)
    ax.grid(True, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['axes.edgecolor'])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['figure.facecolor'])
        print(f"  [Plot] Saved: {save_path}")
    plt.close()


def save_benchmark_json(benchmark_data: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    print(f"  [Benchmark] Saved: {path}")
