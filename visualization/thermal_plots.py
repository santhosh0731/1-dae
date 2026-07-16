"""Thermal Plots Wrapper."""
from visualization.vector_plotter import VectorPlotter

def save_thermal_plot(t, T_mosfet, T_diode, T_inductor, path):
    VectorPlotter.plot_thermal(t, T_mosfet, T_diode, T_inductor, path)
