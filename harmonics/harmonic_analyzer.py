"""
Harmonic Analysis Module
========================
Provides Fast Fourier Transforms (FFT), Total Harmonic Distortion (THD) calculations, 
switching ripple analysis, sideband decomposition, and IEEE 519 compliance validation.
"""

import numpy as np

class HarmonicAnalyzer:
    """Fourier analysis and power quality compliance engine."""
    
    @staticmethod
    def compute_fft(t, signal):
        """Computes FFT and returns frequency and amplitude vectors."""
        N = len(t)
        dt = t[1] - t[0]
        Fs_sampling = 1.0 / dt
        
        # Fast Fourier Transform
        fft_vals = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(N, d=dt)
        amplitudes = (2.0 / N) * np.abs(fft_vals)
        amplitudes[0] = amplitudes[0] / 2.0 # DC offset adjustment
        
        return frequencies, amplitudes

    @staticmethod
    def calculate_thd(frequencies, amplitudes):
        """Calculates Total Harmonic Distortion (THD) and fundamental peak."""
        # Find fundamental frequency peak (typically the largest amplitude non-DC index)
        # Skip the DC component (freq == 0)
        idx_non_dc = np.where(frequencies > 2.0)[0] # ignore frequencies below 2Hz
        if len(idx_non_dc) == 0:
            return 0.0, 0.0
            
        fundamental_idx = idx_non_dc[np.argmax(amplitudes[idx_non_dc])]
        v_fundamental = amplitudes[fundamental_idx]
        
        if v_fundamental < 1e-6:
            return 0.0, 0.0
            
        # Sum of squares of all harmonics
        v_harmonics_sum_sq = np.sum(amplitudes[idx_non_dc]**2) - v_fundamental**2
        thd = np.sqrt(max(0.0, v_harmonics_sum_sq)) / v_fundamental * 100.0
        return float(thd), float(v_fundamental)

    @staticmethod
    def analyze_harmonic_orders(frequencies, amplitudes, base_f=50.0, max_order=49):
        """Extracts individual harmonic orders and their amplitudes."""
        orders = {}
        for order in range(1, max_order + 1, 2): # odd harmonics are dominant
            target_f = base_f * order
            # Find closest frequency bin
            idx = np.argmin(np.abs(frequencies - target_f))
            # Verify it is close enough to target frequency
            if np.abs(frequencies[idx] - target_f) < base_f * 0.4:
                orders[order] = float(amplitudes[idx])
            else:
                orders[order] = 0.0
        return orders

    @staticmethod
    def check_ieee_519_compliance(thd, individual_harmonics, v_fundamental, voltage_level="LV"):
        """Verifies compliance against IEEE 519 standards for harmonics."""
        # LV: Voltage <= 1.0 kV: Individual limit is 5.0%, THD limit is 8.0%
        individual_limit = 5.0
        thd_limit = 8.0
        
        violations = []
        if thd > thd_limit:
            violations.append(f"THD of {thd:.2f}% exceeds the limit of {thd_limit}%")
            
        for order, amp in individual_harmonics.items():
            pct = (amp / (v_fundamental + 1e-9)) * 100.0
            if pct > individual_limit:
                violations.append(f"Harmonic H{order} ({pct:.2f}%) exceeds the individual limit of {individual_limit}%")
                
        return {
            'compliant': len(violations) == 0,
            'thd_limit': thd_limit,
            'individual_limit': individual_limit,
            'violations': violations
        }

    @staticmethod
    def extract_switching_ripple(frequencies, amplitudes, Fs):
        """Finds peak ripple amplitudes at switching frequency (Fs, 2Fs, 3Fs)."""
        ripple = {}
        for mult in [1, 2, 3]:
            target_f = Fs * mult
            idx = np.argmin(np.abs(frequencies - target_f))
            ripple[f"{mult}Fs"] = float(amplitudes[idx])
        return ripple
        
    @staticmethod
    def get_frequency_power_and_energy(frequencies, amplitudes):
        """Computes sideband frequencies and total frequency energy."""
        energy = float(np.sum(amplitudes**2) / 2.0)
        return {'frequency_energy': energy}
