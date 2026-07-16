# Scientific Machine Learning Platform: Mathematical and API Reference

This document provides the mathematical equations, DAE formulations, and API reference guidelines for the modern power electronics modules added to this research framework.

---

## 1. Mathematical Converter Topologies (DAE Formulations)

Each converter is represented as a Differential-Algebraic Equation (DAE) system:

$$F(\mathbf{x}, \mathbf{\dot{x}}, \mathbf{z}, \mathbf{u}, t) = 0$$

Where:
* $\mathbf{x}$ are differential state variables (inductor currents, capacitor voltages).
* $\mathbf{z}$ are algebraic state variables (output node voltage, switching node potential).
* $\mathbf{u}$ are operating inputs ($V_{in}$, switching state $q$, duty cycle $D$).

### Buck Converter Equations
$$\frac{dI_L}{dt} = \frac{D \cdot V_{in} - V_c - I_L(r_{DCR} + r_{ESR})}{L}$$
$$\frac{dV_c}{dt} = \frac{I_L - V_c / R_{load}}{C}$$

### Boost Converter Equations
$$\frac{dI_L}{dt} = \frac{V_{in} - (1-D) V_c - I_L(r_{DCR} + r_{ESR})}{L}$$
$$\frac{dV_c}{dt} = \frac{(1-D) I_L - V_c / R_{load}}{C}$$

---

## 2. Solid-State Transformer (SST) Stage Equations

The SST consists of three cascaded stages:

### Active Front End (AFE) Rectifier Stage
$$\frac{di_{\text{grid}}}{dt} = \frac{v_{\text{grid}}(t) - D_{\text{rect}} v_{dc1}}{L_{\text{afe}}}$$
$$\frac{dv_{dc1}}{dt} = \frac{D_{\text{rect}} i_{\text{grid}} - i_{\text{dab}}}{C_{\text{dc1}}}$$

### Dual Active Bridge (DAB) Stage
Power transferred through the high-frequency transformer is given by:
$$P_{\text{dab}} = \frac{V_{dc1} V_{dc2} \cdot n \cdot \phi (1 - \phi)}{2 L_{\text{leakage}} F_s}$$

### Voltage Source Inverter (VSI) Stage
$$\frac{di_{\text{load}}}{dt} = \frac{D_{\text{inv}} v_{dc2} \sin(\omega t) - v_{\text{load}}}{L_{\text{filter}}}$$
$$\frac{dv_{\text{load}}}{dt} = \frac{i_{\text{load}} - v_{\text{load}} / R_{\text{load}}}{C_{\text{filter}}}$$

---

## 3. Harmonic and Power Quality Calculations

### Total Harmonic Distortion (THD)
$$\text{THD}_{\%_v} = \frac{\sqrt{\sum_{k=2}^{N} V_k^2}}{V_1} \times 100\%$$
Where:
* $V_1$ is the peak amplitude of the fundamental frequency component (50 Hz or 60 Hz).
* $V_k$ represents the amplitude of the $k$-th harmonic order.

### IEEE 519 Low-Voltage limits (≤1.0 kV)
* **Individual Harmonic limit**: 5.0%
* **Total Voltage Distortion (THD) limit**: 8.0%

---

## 4. Multi-Physics Semiconductor & Magnetic Loss Formulations

### Steinmetz Core Loss Equation
$$P_{\text{core}} = k \cdot f^\alpha \cdot B_{\text{pk}}^\beta \cdot V_{\text{core}}$$
Where $k, \alpha, \beta$ are ferrite material parameters.

### Conduction and Switching Losses
$$P_{\text{conduction}} = I_{\text{rms}}^2 \cdot R_{\text{ds(on)}}$$
$$P_{\text{switching}} = \frac{1}{2} V_{\text{block}} I_{\text{avg}} (t_{\text{on}} + t_{\text{off}}) F_s$$

### Thermal Network Model
$$\Delta T_j = T_j - T_{\text{ambient}} = P_{\text{total\_loss}} \cdot R_{\text{th\_ja}}$$
