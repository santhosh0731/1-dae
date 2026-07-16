# Future SST Datasets Generation Specs

This document defines the generation specifications for the Solid-State Transformer (SST) dataset sweeps using LTspice or PLECS simulations.

## Simulated Topologies
1. **Active Front End Rectifier**: Parameterized sweeps over grid voltage amplitude, grid impedance, and duty-cycle/modulation-index.
2. **Dual Active Bridge (DAB)**: High-frequency switching transients at 100 kHz. Outputs: primary/secondary winding currents, transformer leakage voltage.
3. **Inverter Stage**: Output LC filter current and grid current dynamics.

## Target Variables
- `Vac_in`: Input grid voltage (nominal 11 kV)
- `Vdc_h`: High-voltage DC link voltage (nominal 20 kV)
- `Vdc_l`: Low-voltage DC link voltage (nominal 400 V)
- `Igrid`: Current injected into utility grid
