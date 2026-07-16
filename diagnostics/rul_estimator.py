"""
Intelligent Diagnostics - Remaining Useful Life (RUL) Estimator
================================================================
Calculates Arrhenius thermal degradation cycles and estimates remaining hours of operation.
"""

import numpy as np

class RULMultiplier:
    """Estimates capacitor and switch degradation over time."""
    
    @staticmethod
    def estimate_rul(temperatures, ESR_degraded=False, D=0.6):
        """
        Estimates Remaining Useful Life (RUL) in hours.
        Nominal life for MOSFET: 50,000 hours at 80°C junction.
        Nominal life for capacitor: 5,000 hours at 85°C rated core.
        """
        # Rated limits
        T_rated_cap = 85.0
        T_rated_mos = 80.0
        
        T_mos = temperatures.get('mosfet_junction_temp', 25.0)
        T_cap = temperatures.get('inductor_core_temp', 25.0) # assume similar cap temperature
        
        # 1. MOSFET Switch wear out estimation (Arrhenius model)
        # Life is halved for every 10 C rise above 80 C
        delta_T_mos = T_mos - T_rated_mos
        if delta_T_mos > 0:
            life_factor_mos = 2.0 ** (delta_T_mos / 10.0)
            rul_mos = 50000.0 / life_factor_mos
        else:
            rul_mos = 50000.0
            
        # 2. Capacitor wear out (Double Lifetime Law)
        # If ESR is already flagged as degraded, baseline decreases immediately by 50%
        base_cap_life = 2500.0 if ESR_degraded else 5000.0
        delta_T_cap = T_cap - T_rated_cap
        if delta_T_cap > 0:
            life_factor_cap = 2.0 ** (delta_T_cap / 10.0)
            rul_cap = base_cap_life / life_factor_cap
        else:
            rul_cap = base_cap_life
            
        return {
            'mosfet_rul_hours': float(rul_mos),
            'capacitor_rul_hours': float(rul_cap),
            'system_rul_hours': float(min(rul_mos, rul_cap)),
            'mosfet_health_percentage': float(max(0.0, min(100.0, 100.0 * (rul_mos / 50000.0)))),
            'capacitor_health_percentage': float(max(0.0, min(100.0, 100.0 * (rul_cap / 5000.0))))
        }
