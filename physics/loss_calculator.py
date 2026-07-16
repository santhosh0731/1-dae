"""
Physics Loss Modeling Module
============================
Calculates semiconductor switching and conduction losses, Steinmetz core losses, 
ac copper losses (skin effect), parasitics, and dead-time losses.
"""

import numpy as np

class LossCalculator:
    """Computes detailed electrical and thermal losses in converter stages."""
    
    @staticmethod
    def calculate_semiconductor_losses(V_block, I_rms, I_avg, Fs, R_dson=0.08, V_f=0.8, t_sw=50e-9):
        """
        Calculates conduction and switching losses for MOSFETs and diodes.
        t_sw: turn-on + turn-off switching times (seconds)
        """
        # 1. Switch Conduction Loss
        P_cond_switch = (I_rms ** 2) * R_dson
        
        # 2. Switch Switching Loss
        # P_sw = 0.5 * V * I * (t_on + t_off) * Fs
        P_sw_switch = 0.5 * V_block * I_avg * t_sw * Fs
        
        # 3. Diode losses
        P_cond_diode = I_avg * V_f
        P_sw_diode = 0.1 * P_sw_switch # rough estimate for diode reverse recovery
        
        return {
            'mosfet_conduction': float(P_cond_switch),
            'mosfet_switching': float(P_sw_switch),
            'diode_conduction': float(P_cond_diode),
            'diode_switching': float(P_sw_diode),
            'semiconductor_total': float(P_cond_switch + P_sw_switch + P_cond_diode + P_sw_diode)
        }

    @staticmethod
    def calculate_magnetic_losses(Fs, I_ac_rms, I_dc, L, Vol_core=1e-5, B_pk=0.15, R_dc=0.04):
        """
        Steinmetz Core Loss + AC/DC Copper winding losses.
        Steinmetz: P_core = k * f^alpha * B^beta * Vol
        """
        # Steinmetz coefficients for standard ferrite core N87
        k = 12.0
        alpha = 1.3
        beta = 2.5
        
        # Core Loss (W/m^3 to W)
        P_core = k * ((Fs/1e3)**alpha) * (B_pk**beta) * Vol_core * 1e3
        
        # Copper Loss: include simple AC skin effect multiplier (frequencies above 10kHz)
        skin_effect = 1.0
        if Fs > 10000.0:
            skin_effect = 1.0 + 0.15 * np.log10(Fs / 10000.0)
            
        R_ac = R_dc * skin_effect
        P_cu = (I_dc**2) * R_dc + (I_ac_rms**2) * R_ac
        
        return {
            'core_loss': float(P_core),
            'copper_loss': float(P_cu),
            'inductor_total': float(P_core + P_cu)
        }

    @staticmethod
    def calculate_capacitor_losses(I_cap_rms, ESR=0.02):
        """ESR losses in filter capacitors."""
        P_cap = (I_cap_rms ** 2) * ESR
        return {'capacitor_loss': float(P_cap)}

    @staticmethod
    def calculate_dead_time_loss(IL, V_diode=0.8, t_dead=100e-9, Fs=50000.0):
        """Losses incurred by body diode conduction during dead time."""
        P_dead = 2 * V_diode * np.abs(IL) * t_dead * Fs
        return {'dead_time_loss': float(P_dead)}
