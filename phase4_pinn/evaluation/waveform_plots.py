"""
Waveform Plots — PINN vs LTspice
===================================
Publication-quality dark-theme plots comparing PINN predictions
against LTspice ground truth waveforms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Dict


DARK_BG    = '#0d1117'
GRID_COLOR = '#21262d'
COLORS = {
    'ltspice': '#58a6ff',
    'pinn':    '#f78166',
    'residual':'#3fb950',
    'phase3':  '#d29922',
}


def plot_waveform_comparison(
    t:         np.ndarray,     # (T,) time axis
    Vout_true: np.ndarray,     # (T,)
    IL_true:   np.ndarray,     # (T,)
    Vout_pred: np.ndarray,     # (T,)
    IL_pred:   np.ndarray,     # (T,)
    title:     str = "PINN vs LTspice",
    save_path: Optional[str] = None,
    params:    Optional[Dict] = None,
) -> None:
    """4-panel waveform comparison plot."""
    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_Vout     = fig.add_subplot(gs[0, 0])
    ax_IL       = fig.add_subplot(gs[0, 1])
    ax_res_Vout = fig.add_subplot(gs[1, 0])
    ax_res_IL   = fig.add_subplot(gs[1, 1])
    ax_scatter  = fig.add_subplot(gs[2, :])

    for ax in [ax_Vout, ax_IL, ax_res_Vout, ax_res_IL, ax_scatter]:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5)

    t_ms = t * 1e3   # convert to ms

    # Vout waveform
    ax_Vout.plot(t_ms, Vout_true, color=COLORS['ltspice'], lw=1.8, label='LTspice GT')
    ax_Vout.plot(t_ms, Vout_pred, color=COLORS['pinn'],    lw=1.4, ls='--', label='PINN Pred')
    ax_Vout.set_title('Output Voltage Vout(t)')
    ax_Vout.set_xlabel('Time [ms]')
    ax_Vout.set_ylabel('Vout [V]')
    ax_Vout.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor='#e6edf3', fontsize=9)

    # IL waveform
    ax_IL.plot(t_ms, IL_true, color=COLORS['ltspice'], lw=1.8, label='LTspice GT')
    ax_IL.plot(t_ms, IL_pred, color=COLORS['pinn'],    lw=1.4, ls='--', label='PINN Pred')
    ax_IL.set_title('Inductor Current IL(t)')
    ax_IL.set_xlabel('Time [ms]')
    ax_IL.set_ylabel('IL [A]')
    ax_IL.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor='#e6edf3', fontsize=9)

    # Residuals
    ax_res_Vout.plot(t_ms, Vout_pred - Vout_true, color=COLORS['residual'], lw=1.2)
    ax_res_Vout.axhline(0, color='#888', lw=0.5)
    ax_res_Vout.set_title('Vout Residual (Pred - GT)')
    ax_res_Vout.set_xlabel('Time [ms]')
    ax_res_Vout.set_ylabel('Error [V]')

    ax_res_IL.plot(t_ms, IL_pred - IL_true, color=COLORS['residual'], lw=1.2)
    ax_res_IL.axhline(0, color='#888', lw=0.5)
    ax_res_IL.set_title('IL Residual (Pred - GT)')
    ax_res_IL.set_xlabel('Time [ms]')
    ax_res_IL.set_ylabel('Error [A]')

    # Scatter true vs pred
    all_true = np.concatenate([Vout_true / Vout_true.max(), IL_true / (IL_true.max() + 1e-9)])
    all_pred = np.concatenate([Vout_pred / Vout_true.max(), IL_pred / (IL_true.max() + 1e-9)])
    ax_scatter.scatter(all_true, all_pred, s=3, alpha=0.4, color=COLORS['pinn'])
    lims = [min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())]
    ax_scatter.plot(lims, lims, 'w--', lw=1, alpha=0.5)
    ax_scatter.set_title('True vs Predicted (normalized)')
    ax_scatter.set_xlabel('True')
    ax_scatter.set_ylabel('Predicted')

    # Parameter annotation
    if params:
        txt = (f"Vin={params.get('Vin','?')}V  D={params.get('D','?')}  "
               f"Fs={params.get('Fs','?')/1e3:.0f}kHz  "
               f"L={params.get('L','?')*1e6:.0f}µH  "
               f"C={params.get('C','?')*1e6:.0f}µF  "
               f"Rload={params.get('Rload','?')}Ω")
        fig.text(0.5, 0.01, txt, ha='center', fontsize=9, color='#8b949e')

    fig.suptitle(title, color='#e6edf3', fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  [Plot] Saved: {save_path}")
    plt.close()


def plot_loss_curves(
    history: Dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot training and physics loss curves per epoch."""
    epochs = history.get('epoch', [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=DARK_BG)
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors='#8b949e')
        ax.grid(True, color=GRID_COLOR, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    # Total loss
    axes[0].semilogy(epochs, history.get('train_total', []), color=COLORS['ltspice'], label='Train')
    axes[0].semilogy(epochs, history.get('val_total',   []), color=COLORS['pinn'],    label='Val', ls='--')
    axes[0].set_title('Total Loss', color='#e6edf3')
    axes[0].legend(facecolor=DARK_BG, labelcolor='#e6edf3')

    # Data loss
    axes[1].semilogy(epochs, history.get('data', []), color='#79c0ff', label='Data Loss')
    axes[1].set_title('Data Loss', color='#e6edf3')

    # Physics losses
    for key, col, lab in [('kvl','#f78166','KVL'), ('kcl','#3fb950','KCL'), ('dae','#d29922','DAE')]:
        if history.get(key):
            axes[2].semilogy(epochs, history[key], color=col, label=lab)
    axes[2].set_title('Physics Losses', color='#e6edf3')
    axes[2].legend(facecolor=DARK_BG, labelcolor='#e6edf3')

    # Lambda weights
    if history.get('lambda_kvl'):
        axes[3].plot(epochs, history['lambda_kvl'], color=COLORS['pinn'],    label='λ_KVL')
        axes[3].plot(epochs, history['lambda_kcl'], color=COLORS['residual'], label='λ_KCL')
    axes[3].set_title('Adaptive λ Weights', color='#e6edf3')
    axes[3].legend(facecolor=DARK_BG, labelcolor='#e6edf3')

    for ax in axes:
        ax.set_xlabel('Epoch', color='#8b949e')

    fig.suptitle('PINN Training History', color='#e6edf3', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  [Plot] Saved: {save_path}")
    plt.close()
