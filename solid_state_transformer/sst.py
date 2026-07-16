"""
Solid-State Transformer (SST) Module
====================================
Modular modeling of a three-stage SST: AFE Rectifier, HV DC Link, Dual Active Bridge (DAB) 
with High-Frequency Transformer, LV DC Link, Inverter, and load side.
"""

import numpy as np
from scipy.integrate import solve_ivp

class AFERectifierStage:
    """Stage 1: Active Front End Rectifier (Grid to HV DC Link)"""
    def __init__(self, L=1.5e-3, C=470e-6, R=500.0):
        self.L = L
        self.C = C
        self.R = R # Equivalent load resistance of DC link

    def get_derivatives(self, t, x, Vin, D, grid_f):
        # x = [i_grid, v_dc1]
        i_grid, v_dc1 = x[0], x[1]
        w = 2 * np.pi * grid_f
        v_grid = Vin * np.sin(w * t)
        
        di_grid_dt = (v_grid - D * v_dc1) / self.L
        dv_dc1_dt = (D * i_grid - v_dc1 / self.R) / self.C
        return np.array([di_grid_dt, dv_dc1_dt])

class DABStage:
    """Stage 2: Dual Active Bridge Converter (HV DC Link to LV DC Link)"""
    def __init__(self, L_leakage=10e-6, C_lv=2200e-6, n=10.0):
        self.L_leakage = L_leakage
        self.C_lv = C_lv
        self.n = n # HF Transformer turns ratio (HV to LV)

    def get_derivatives(self, t, x, v_dc1, v_dc2, phi, Fs):
        # x = [i_leakage]
        # DAB average power equation
        # phi is phase shift in fraction of pi (0 to 0.5)
        P_trans = (v_dc1 * v_dc2 * self.n * phi * (1.0 - phi)) / (2.0 * self.L_leakage * Fs)
        
        i_leakage = x[0]
        di_leakage_dt = (v_dc1 - v_dc2 * self.n) / self.L_leakage
        return np.array([di_leakage_dt]), P_trans

class InverterStage:
    """Stage 3: Voltage Source Inverter (LV DC Link to Load)"""
    def __init__(self, L_filter=2e-3, C_filter=100e-6):
        self.L = L_filter
        self.C = C_filter

    def get_derivatives(self, t, x, v_dc2, D, load_f, Rload):
        # x = [i_load, v_load]
        i_load, v_load = x[0], x[1]
        w = 2 * np.pi * load_f
        v_inverter = D * v_dc2 * np.sin(w * t)
        
        di_load_dt = (v_inverter - v_load) / self.L
        dv_load_dt = (i_load - v_load / Rload) / self.C
        return np.array([di_load_dt, dv_load_dt])

class SolidStateTransformerModel:
    """Complete 3-Stage Solid-State Transformer (SST) System"""
    name = "Solid-State Transformer"
    state_names = ['i_grid', 'v_dc_hv', 'i_leakage', 'v_dc_lv', 'i_load', 'v_load']
    
    def __init__(self):
        self.afe = AFERectifierStage()
        self.dab = DABStage()
        self.inverter = InverterStage()

    def get_derivatives(self, t, x, params):
        # x = [i_grid, v_dc_hv, i_leakage, v_dc_lv, i_load, v_load]
        i_grid, v_dc_hv, i_leakage, v_dc_lv, i_load, v_load = x[0], x[1], x[2], x[3], x[4], x[5]
        
        Vin = params.get('Vin', 110.0) # Input AC Amplitude
        D = params.get('D', 0.6)       # Modulation ratio
        Fs = params.get('Fs', 50000.0)  # Switching frequency
        Rload = params.get('Rload', 10.0)
        grid_f = params.get('grid_frequency', 50.0)
        
        # 1. AFE Stage
        d_afe = self.afe.get_derivatives(t, [i_grid, v_dc_hv], Vin, D, grid_f)
        
        # 2. DAB Stage
        phi = params.get('phi_shift', 0.15) # Phase shift angle
        d_dab, P_trans = self.dab.get_derivatives(t, [i_leakage], v_dc_hv, v_dc_lv, phi, Fs)
        
        # Coupling HV link to DAB load
        d_afe[1] -= (P_trans / (v_dc_hv + 1e-9)) / self.afe.C
        
        # Coupling LV link to DAB feed and Inverter feed
        i_inverter_in = D * i_load # inverter input current from DC link
        dv_dc_lv_dt = ((P_trans / (v_dc_lv + 1e-9)) - i_inverter_in) / self.dab.C_lv
        
        # 3. Inverter Stage
        d_inv = self.inverter.get_derivatives(t, [i_load, v_load], v_dc_lv, D, grid_f, Rload)
        
        return np.array([
            d_afe[0],   # di_grid_dt
            d_afe[1],   # dv_dc_hv_dt
            d_dab[0],   # di_leakage_dt
            dv_dc_lv_dt, # dv_dc_lv_dt
            d_inv[0],   # di_load_dt
            d_inv[1]    # dv_load_dt
        ])

    def solve_dae(self, t_span, y0, t_eval, params):
        """Solves the SST combined ODE system using Radau solver."""
        def odefun(t, y):
            return self.get_derivatives(t, y, params)
            
        sol = solve_ivp(
            odefun,
            t_span,
            y0,
            t_eval=t_eval,
            method='Radau',
            rtol=1e-5,
            atol=1e-7
        )
        return sol.t, sol.y.T

    def get_surrogate_prediction(self):
        """Unified interface returning unavailable if no specific SST neural model is loaded."""
        return "Model unavailable"
