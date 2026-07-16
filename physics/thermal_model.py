"""
Thermal Modeling Module
=======================
Calculates temperature rise in semiconductor junctions and inductor cores.
"""

class ThermalModel:
    """Thermal equivalent model solver."""
    
    @staticmethod
    def calculate_temperatures(P_mosfet, P_diode, P_inductor, T_ambient=25.0, Rth_j_amb=62.5, Rth_core_amb=35.0):
        """
        Computes temperatures based on thermal resistance of packages.
        Rth_j_amb: Junction-to-ambient thermal resistance (C/W)
        Rth_core_amb: Core-to-ambient thermal resistance (C/W)
        """
        T_mosfet_j = T_ambient + P_mosfet * Rth_j_amb
        T_diode_j = T_ambient + P_diode * Rth_j_amb
        T_inductor_core = T_ambient + P_inductor * Rth_core_amb
        
        return {
            'mosfet_junction_temp': float(T_mosfet_j),
            'diode_junction_temp': float(T_diode_j),
            'inductor_core_temp': float(T_inductor_core),
            'max_temp': float(max(T_mosfet_j, T_diode_j, T_inductor_core))
        }
