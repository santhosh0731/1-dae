"""
Grid & Renewable Energy - Grid Events
=====================================
Simulates power quality disturbances: voltage sag, voltage swell, frequency drift, and load steps.
"""

import numpy as np

class GridEvents:
    """Modulates grid voltage and load parameters to represent grid anomalies."""
    
    @staticmethod
    def get_voltage_profile(t, Vin_nominal, event_type='none', depth_or_swell=0.3, t_start=0.001, t_duration=0.002):
        """Returns time-varying grid voltage Vin(t)."""
        Vin_t = np.full_like(t, Vin_nominal)
        
        if event_type == 'sag':
            mask = (t >= t_start) & (t <= (t_start + t_duration))
            Vin_t[mask] = Vin_nominal * (1.0 - depth_or_swell)
        elif event_type == 'swell':
            mask = (t >= t_start) & (t <= (t_start + t_duration))
            Vin_t[mask] = Vin_nominal * (1.0 + depth_or_swell)
            
        return Vin_t

    @staticmethod
    def get_load_profile(t, Rload_nominal, event_type='none', step_ratio=0.5, t_step=0.002):
        """Returns time-varying load resistance Rload(t) to represent a step load change."""
        Rload_t = np.full_like(t, Rload_nominal)
        
        if event_type == 'step':
            mask = t >= t_step
            Rload_t[mask] = Rload_nominal * step_ratio
            
        return Rload_t

    @staticmethod
    def get_frequency_profile(t, f_nominal, drift_hz=5.0, rate_hz_sec=100.0):
        """Returns time-varying frequency f(t) representing frequency drifts."""
        # Simple ramp drift
        f_t = f_nominal + drift_hz * (1.0 - np.exp(-t * rate_hz_sec))
        return f_t
