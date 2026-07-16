"""Sensitivity Plots Wrapper."""
from visualization.vector_plotter import VectorPlotter

def save_sensitivity_plot(sensitivities, path):
    VectorPlotter.plot_sensitivity(sensitivities, path)
