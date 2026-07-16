"""Benchmark Plots Wrapper."""
import matplotlib.pyplot as plt
from visualization.vector_plotter import VectorPlotter

def save_benchmark_plot(benchmarks, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    models = list(benchmarks.keys())
    # Plot R2 if numeric
    r2_vals = [benchmarks[m]['r2'] if isinstance(benchmarks[m]['r2'], (int, float)) else 0.0 for m in models]
    ax.bar(models, r2_vals, color='orange', edgecolor='black')
    ax.set_ylabel('R2 Score')
    ax.set_title('Neural Surrogate Accuracy Comparison')
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    fig.tight_layout()
    VectorPlotter.save_figure(fig, path)
