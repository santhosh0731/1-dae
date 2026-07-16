"""FFT Plots Wrapper."""
from visualization.vector_plotter import VectorPlotter

def save_fft_plot(freqs, amps, path):
    VectorPlotter.plot_fft(freqs, amps, path)
