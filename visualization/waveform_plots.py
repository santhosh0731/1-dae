"""Waveform Plots Wrapper."""
from visualization.vector_plotter import VectorPlotter

def save_waveform_plot(t, Vout, IL, path):
    VectorPlotter.plot_waveforms(t, Vout, IL, path)
