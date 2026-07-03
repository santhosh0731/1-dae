"""
Digital Twin Simulator
=======================
Loads the deployment TorchScript DAE-PINN model, simulates transient states
for customized user inputs, and saves dashboard plots.
"""

import os
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

DARK_BG = '#0d1117'
GRID_COLOR = '#21262d'


class DigitalTwinDashboard:
    """Real-time simulation dashboard wrapper for DAE-PINN."""

    def __init__(self, model_path: Path, scalers_path: Path, output_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_simulation(
        self,
        Vin: float,
        D: float,
        Fs: float,
        L: float,
        C: float,
        Rload: float,
        t_end: float = 0.005,
        steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run high-speed inference for custom converter parameters.
        Returns:
            t: time axis (seconds)
            pred_real: array of shape (steps, 3) [Vout, IL, Vc]
        """
        t_eval = np.linspace(0, t_end, steps, dtype=np.float32)

        # Build raw input grid
        raw_inputs = np.column_stack([
            t_eval,
            np.full(steps, Vin,   dtype=np.float32),
            np.full(steps, D,     dtype=np.float32),
            np.full(steps, Fs,    dtype=np.float32),
            np.full(steps, L,     dtype=np.float32),
            np.full(steps, C,     dtype=np.float32),
            np.full(steps, Rload, dtype=np.float32),
        ])

        # Normalize inputs
        norm_inputs = self.scalers['X'].transform(raw_inputs).astype(np.float32)
        X_tensor = torch.tensor(norm_inputs, device=self.device)

        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()

        # Denormalize predictions (predictions are already physical)
        pred_real = pred[:, :3]

        return t_eval, pred_real

    def plot_dashboard(
        self,
        t: np.ndarray,
        preds: np.ndarray,
        params: Dict[str, float],
        filename: str = "digital_twin_telemetry.png",
    ) -> Path:
        """Plot telemetry dashboard of simulated states."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor=DARK_BG)

        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors='#8b949e')
            ax.grid(True, color=GRID_COLOR, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)

        t_ms = t * 1e3
        Vout = preds[:, 0]
        IL = preds[:, 1]

        # 1. Output Voltage plot
        axes[0].plot(t_ms, Vout, color='#f78166', lw=2, label='DAE-PINN Vout')
        axes[0].set_title('Output Voltage Vout(t) Telemetry', color='#e6edf3')
        axes[0].set_ylabel('Voltage [V]', color='#8b949e')

        # 2. Inductor Current plot
        axes[1].plot(t_ms, IL, color='#58a6ff', lw=2, label='DAE-PINN IL')
        axes[1].set_title('Inductor Current IL(t) Telemetry', color='#e6edf3')
        axes[1].set_xlabel('Time [ms]', color='#8b949e')
        axes[1].set_ylabel('Current [A]', color='#8b949e')

        title = (f"Digital Twin Live Telemetry: Vin={params['Vin']}V | D={params['D']} | "
                 f"Fs={params['Fs']/1e3:.0f}kHz | Rload={params['Rload']}Ω")
        fig.suptitle(title, color='#e6edf3', fontsize=12, fontweight='bold')
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150, facecolor=DARK_BG)
        plt.close()
        return out_path
