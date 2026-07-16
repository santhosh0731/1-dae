"""
Advanced Visualization - Vector Plotter
========================================
Generates publication-quality figure layouts (SVG, high-resolution 600 DPI PNG, PDF).
"""

import matplotlib
matplotlib.use('Agg') # Thread-safe non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Scientific journal figure configurations
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
})

class VectorPlotter:
    """Renders high-resolution vector and raster assets."""
    
    @staticmethod
    def save_figure(fig, output_path: str):
        """Saves a figure as both vector SVG and 600 DPI PNG."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vector SVG
        fig.savefig(path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        # Save 600 DPI PNG
        fig.savefig(path.with_suffix('.png'), format='png', dpi=600, bbox_inches='tight')
        
        plt.close(fig)

    @staticmethod
    def plot_waveforms(t, Vout, IL, output_path: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4.5))
        
        ax1.plot(t * 1000, Vout, label=r'$V_{\mathrm{out}}(t)$', color='#007acc', linewidth=1.5)
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_title('Transient Waveform Simulation Responses')
        
        ax2.plot(t * 1000, IL, label=r'$I_L(t)$', color='#ff6600', linewidth=1.5)
        ax2.set_ylabel('Current (A)')
        ax2.set_xlabel('Time (ms)')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        
        fig.tight_layout()
        VectorPlotter.save_figure(fig, output_path)

    @staticmethod
    def plot_fft(frequencies, amplitudes, output_path: str):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(frequencies[:60], amplitudes[:60], color='#4b0082', width=200.0)
        ax.set_ylabel('Amplitude (V)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Harmonic Power Spectrum (FFT)')
        ax.grid(True)
        
        fig.tight_layout()
        VectorPlotter.save_figure(fig, output_path)

    @staticmethod
    def plot_sensitivity(sensitivities: dict, output_path: str):
        fig, ax = plt.subplots(figsize=(6, 3))
        keys = list(sensitivities.keys())
        values = list(sensitivities.values())
        
        colors = ['#ff4f4f' if v < 0 else '#4fff4f' for v in values]
        ax.barh(keys, values, color=colors, edgecolor='grey', height=0.5)
        ax.set_xlabel('Normalized Sensitivity Index')
        ax.set_title('Dynamic Sensitivity Parameter Ranking')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True)
        
        fig.tight_layout()
        VectorPlotter.save_figure(fig, output_path)

    @staticmethod
    def plot_thermal(t, T_mosfet, T_diode, T_inductor, output_path: str):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t * 1000, T_mosfet, label='MOSFET', color='red', linewidth=1.2)
        ax.plot(t * 1000, T_diode, label='Diode', color='blue', linewidth=1.2)
        ax.plot(t * 1000, T_inductor, label='Inductor', color='green', linewidth=1.2)
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Multi-Physics Junction Thermal Profile')
        ax.grid(True)
        ax.legend()
        
        fig.tight_layout()
        VectorPlotter.save_figure(fig, output_path)
