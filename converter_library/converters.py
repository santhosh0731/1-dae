"""
Universal Converter Library
===========================
Mathematical equations, state variables, and DAE formulations for 19 power converter topologies.
"""

import numpy as np
from scipy.integrate import solve_ivp

class BaseConverter:
    """Base class for converter models."""
    name = "Base Converter"
    state_names = []
    param_names = ['Vin', 'D', 'Fs', 'L', 'C', 'Rload']
    
    def get_derivatives(self, t, x, params):
        raise NotImplementedError
        
    def get_algebraic_constraints(self, x, params):
        # Default: Vout equals Vc
        Vout = x[1] if len(x) > 1 else 0.0
        Vc = x[1] if len(x) > 1 else 0.0
        return np.array([Vout - Vc])

    def solve_dae(self, t_span, y0, t_eval, params):
        """Solves the state equations using SciPy's stiff Radau solver."""
        # Unpack parameters
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        Fs = params.get('Fs', 50000.0)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        
        # Resolve parasitic values
        ESR = params.get('ESR', 0.02)
        DCR = params.get('DCR', 0.05)
        
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

class BuckConverter(BaseConverter):
    name = "Buck Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        DCR = params.get('DCR', 0.05)
        ESR = params.get('ESR', 0.02)
        
        # Average mode equations with parasitical resistance
        dIL_dt = (D * Vin - Vc - IL * (DCR + ESR)) / L
        dVc_dt = (IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class BoostConverter(BaseConverter):
    name = "Boost Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        DCR = params.get('DCR', 0.05)
        ESR = params.get('ESR', 0.02)
        
        dIL_dt = (Vin - (1.0 - D) * Vc - IL * (DCR + ESR)) / L
        dVc_dt = ((1.0 - D) * IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class BuckBoostConverter(BaseConverter):
    name = "Buck-Boost Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        DCR = params.get('DCR', 0.05)
        
        dIL_dt = (D * Vin + (1.0 - D) * Vc - IL * DCR) / L
        dVc_dt = (-(1.0 - D) * IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class CukConverter(BaseConverter):
    name = "Ćuk Converter"
    state_names = ['IL1', 'IL2', 'Vc1', 'Vc2']
    
    def get_derivatives(self, t, x, params):
        IL1, IL2, Vc1, Vc2 = x[0], x[1], x[2], x[3]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)  # Using L as L1, L2 is scaled
        L1, L2 = L, L * 1.2
        C = params.get('C', 47e-6)  # Using C as C1, C2 is scaled
        C1, C2 = C, C * 1.5
        Rload = params.get('Rload', 5.0)
        
        dIL1_dt = (Vin - (1.0 - D) * Vc1) / L1
        dIL2_dt = (D * Vc1 - Vc2) / L2
        dVc1_dt = ((1.0 - D) * IL1 - D * IL2) / C1
        dVc2_dt = (IL2 - Vc2 / Rload) / C2
        return np.array([dIL1_dt, dIL2_dt, dVc1_dt, dVc2_dt])

class SepicConverter(BaseConverter):
    name = "SEPIC Converter"
    state_names = ['IL1', 'IL2', 'Vc1', 'Vc2']
    
    def get_derivatives(self, t, x, params):
        IL1, IL2, Vc1, Vc2 = x[0], x[1], x[2], x[3]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        L1, L2 = L, L * 1.1
        C = params.get('C', 47e-6)
        C1, C2 = C, C * 1.3
        Rload = params.get('Rload', 5.0)
        
        dIL1_dt = (Vin - (1.0 - D) * Vc1 - (1.0 - D) * Vc2) / L1
        dIL2_dt = (D * Vc1 - (1.0 - D) * Vc2) / L2
        dVc1_dt = ((1.0 - D) * IL1 - D * IL2) / C1
        dVc2_dt = ((1.0 - D) * IL1 + (1.0 - D) * IL2 - Vc2 / Rload) / C2
        return np.array([dIL1_dt, dIL2_dt, dVc1_dt, dVc2_dt])

class ZetaConverter(BaseConverter):
    name = "Zeta Converter"
    state_names = ['IL1', 'IL2', 'Vc1', 'Vc2']
    
    def get_derivatives(self, t, x, params):
        IL1, IL2, Vc1, Vc2 = x[0], x[1], x[2], x[3]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        L1, L2 = L, L * 1.1
        C = params.get('C', 47e-6)
        C1, C2 = C, C * 1.3
        Rload = params.get('Rload', 5.0)
        
        dIL1_dt = (D * Vin - (1.0 - D) * Vc1) / L1
        dIL2_dt = (D * Vin + (1.0 - D) * Vc1 - Vc2) / L2
        dVc1_dt = ((1.0 - D) * IL1 - D * IL2) / C1
        dVc2_dt = (IL2 - Vc2 / Rload) / C2
        return np.array([dIL1_dt, dIL2_dt, dVc1_dt, dVc2_dt])

class LuoConverter(BaseConverter):
    name = "Luo Converter"
    state_names = ['IL1', 'Vc1', 'Vc2']
    
    def get_derivatives(self, t, x, params):
        IL1, Vc1, Vc2 = x[0], x[1], x[2]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        C1, C2 = C, C * 1.2
        Rload = params.get('Rload', 5.0)
        
        dIL1_dt = (Vin - (1.0 - D) * Vc1 - (1.0 - D) * Vc2) / L
        dVc1_dt = (IL1 - (1.0 - D) * Vc1 / Rload) / C1
        dVc2_dt = (IL1 - Vc2 / Rload) / C2
        return np.array([dIL1_dt, dVc1_dt, dVc2_dt])

class FlybackConverter(BaseConverter):
    name = "Flyback Converter"
    state_names = ['Im', 'Vc']
    
    def get_derivatives(self, t, x, params):
        Im, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 1.0) # Transformer turns ratio
        
        dIm_dt = (D * Vin - (1.0 - D) * Vc * Np_Ns) / L
        dVc_dt = ((1.0 - D) * Im * Np_Ns - Vc / Rload) / C
        return np.array([dIm_dt, dVc_dt])

class ForwardConverter(BaseConverter):
    name = "Forward Converter"
    state_names = ['Im', 'IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        Im, IL, Vc = x[0], x[1], x[2]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        Lm = L * 10.0 # Magnetizing inductance
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 2.0)
        
        dIm_dt = (D * Vin - (1.0 - D) * Im) / Lm
        dIL_dt = (D * Vin / Np_Ns - Vc) / L
        dVc_dt = (IL - Vc / Rload) / C
        return np.array([dIm_dt, dIL_dt, dVc_dt])

class PushPullConverter(BaseConverter):
    name = "Push-Pull Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6) # Switch duty cycle, D < 0.5
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 2.0)
        
        dIL_dt = (2 * D * Vin / Np_Ns - Vc) / L
        dVc_dt = (IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class HalfBridgeConverter(BaseConverter):
    name = "Half-Bridge Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6) # D < 0.5
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 1.0)
        
        dIL_dt = (D * Vin / Np_Ns - Vc) / L
        dVc_dt = (IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class FullBridgeConverter(BaseConverter):
    name = "Full-Bridge Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 1.0)
        
        dIL_dt = (D * Vin / Np_Ns - Vc) / L
        dVc_dt = (IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class LlcResonantConverter(BaseConverter):
    name = "LLC Resonant Converter"
    state_names = ['Ir', 'Im', 'Vc']
    
    def get_derivatives(self, t, x, params):
        Ir, Im, Vc = x[0], x[1], x[2]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6) # Resonant inductance Lr
        Lm = L * 4.0 # Magnetizing inductance Lm
        C = params.get('C', 47e-6) # Resonant capacitance Cr, using C
        Co = C * 5.0 # Output filter capacitance
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 1.0)
        
        # LLC Simplified average state equations
        dIr_dt = (Vin * D - Vc * Np_Ns) / L
        dIm_dt = (Vc * Np_Ns) / Lm
        dVc_dt = (Ir - Im / Rload) / Co
        return np.array([dIr_dt, dIm_dt, dVc_dt])

class InterleavedBoostConverter(BaseConverter):
    name = "Interleaved Boost Converter"
    state_names = ['IL1', 'IL2', 'Vc']
    
    def get_derivatives(self, t, x, params):
        IL1, IL2, Vc = x[0], x[1], x[2]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        
        dIL1_dt = (Vin - (1.0 - D) * Vc) / L
        dIL2_dt = (Vin - (1.0 - D) * Vc) / L
        dVc_dt = (((1.0 - D) * IL1 + (1.0 - D) * IL2) - Vc / Rload) / C
        return np.array([dIL1_dt, dIL2_dt, dVc_dt])

class BidirectionalDcDcConverter(BaseConverter):
    name = "Bidirectional DC/DC Converter"
    state_names = ['IL', 'Vc']
    
    def get_derivatives(self, t, x, params):
        # Behaves as Buck in forward direction, Boost in reverse
        IL, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        mode = params.get('operating_mode', 'Buck') # 'Buck' or 'Boost'
        
        if mode == 'Buck':
            dIL_dt = (D * Vin - Vc) / L
            dVc_dt = (IL - Vc / Rload) / C
        else:
            dIL_dt = (Vin - (1.0 - D) * Vc) / L
            dVc_dt = ((1.0 - D) * IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

class DualActiveBridgeConverter(BaseConverter):
    name = "Dual Active Bridge Converter"
    state_names = ['Ip', 'Vc']
    
    def get_derivatives(self, t, x, params):
        Ip, Vc = x[0], x[1]
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6) # Represents phase shift angle phi
        L = params.get('L', 50e-6) # Leakage inductance
        C = params.get('C', 47e-6) # Output cap
        Rload = params.get('Rload', 5.0)
        Np_Ns = params.get('Np_Ns', 1.0)
        
        # DAB average power equation used to solve state updates
        Power_DAB = (Vin * Vc * Np_Ns * D * (1.0 - D)) / (2 * L * 50000.0) # at Fs=50kHz
        dIp_dt = (Vin - Vc / Np_Ns) / L
        dVc_dt = (Power_DAB / (Vc + 1e-9) - Vc / Rload) / C
        return np.array([dIp_dt, dVc_dt])

class ActiveFrontEndConverter(BaseConverter):
    name = "Active Front End"
    state_names = ['i_grid', 'V_dc']
    
    def get_derivatives(self, t, x, params):
        i_grid, V_dc = x[0], x[1]
        Vin = params.get('Vin', 48.0) # V_grid amplitude
        D = params.get('D', 0.6)      # modulation index
        L = params.get('L', 50e-6)      # Grid inductor L
        C = params.get('C', 47e-6)      # DC Link Cap C
        Rload = params.get('Rload', 5.0)
        w = 2 * np.pi * params.get('grid_frequency', 50.0)
        
        v_grid = Vin * np.sin(w * t)
        dI_grid_dt = (v_grid - D * V_dc) / L
        dV_dc_dt = (D * i_grid - V_dc / Rload) / C
        return np.array([dI_grid_dt, dV_dc_dt])

class VoltageSourceInverter(BaseConverter):
    name = "Voltage Source Inverter"
    state_names = ['i_load', 'v_load']
    
    def get_derivatives(self, t, x, params):
        i_load, v_load = x[0], x[1]
        Vin = params.get('Vin', 48.0) # DC Link input voltage
        D = params.get('D', 0.6)      # Modulation index
        L = params.get('L', 50e-6)      # filter L
        C = params.get('C', 47e-6)      # filter C
        Rload = params.get('Rload', 5.0)
        w = 2 * np.pi * params.get('grid_frequency', 50.0)
        
        v_inverter = D * Vin * np.sin(w * t)
        di_load_dt = (v_inverter - v_load) / L
        dv_load_dt = (i_load - v_load / Rload) / C
        return np.array([di_load_dt, dv_load_dt])

# Registry for universal lookup
CONVERTER_REGISTRY = {
    'buck': BuckConverter(),
    'boost': BoostConverter(),
    'buck_boost': BuckBoostConverter(),
    'cuk': CukConverter(),
    'sepic': SepicConverter(),
    'zeta': ZetaConverter(),
    'luo': LuoConverter(),
    'flyback': FlybackConverter(),
    'forward': ForwardConverter(),
    'push_pull': PushPullConverter(),
    'half_bridge': HalfBridgeConverter(),
    'full_bridge': FullBridgeConverter(),
    'llc_resonant': LlcResonantConverter(),
    'interleaved_boost': InterleavedBoostConverter(),
    'bidirectional_dc_dc': BidirectionalDcDcConverter(),
    'dual_active_bridge': DualActiveBridgeConverter(),
    'active_front_end': ActiveFrontEndConverter(),
    'voltage_source_inverter': VoltageSourceInverter()
}
