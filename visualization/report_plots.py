"""Report Plots Wrapper."""
import matplotlib.pyplot as plt
from visualization.vector_plotter import VectorPlotter

def save_report_summary_plot(losses, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    keys = ['MOSFET', 'Inductor', 'Capacitor', 'Dead Time']
    vals = [
        losses['semiconductor']['semiconductor_total'],
        losses['magnetic']['inductor_total'],
        losses['capacitor']['capacitor_loss'],
        losses['dead_time']['dead_time_loss']
    ]
    ax.pie(vals, labels=keys, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    ax.set_title('Platform Loss Summary Distribution')
    fig.tight_layout()
    VectorPlotter.save_figure(fig, path)
