"""Optimization Plots Wrapper."""
import matplotlib.pyplot as plt
from visualization.vector_plotter import VectorPlotter

def save_optimization_plot(sweep_results, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = [r['value'] for r in sweep_results]
    y = [r['vout_avg'] for r in sweep_results]
    ax.plot(x, y, marker='o', color='teal', linewidth=2.0)
    ax.set_xlabel('Duty Cycle (D)')
    ax.set_ylabel('Average Output Voltage (V)')
    ax.set_title('Optimization Parameter Sweep Curve')
    ax.grid(True)
    fig.tight_layout()
    VectorPlotter.save_figure(fig, path)
