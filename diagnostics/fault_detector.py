"""
Intelligent Diagnostics - Fault Detector
=========================================
Rule-based and wave-informed circuit fault classifier.
"""

import numpy as np

class FaultDetector:
    """Classifies standard power converter faults based on waveform states."""
    
    @staticmethod
    def classify_faults(t, Vout, IL, temperatures, params):
        """Identifies active faults like open/short circuit, degraded ESR, or saturation."""
        faults = []
        prob = 0.0
        
        Vin = params.get('Vin', 48.0)
        D = params.get('D', 0.6)
        ESR = params.get('ESR', 0.02)
        L = params.get('L', 50e-6)
        C = params.get('C', 47e-6)
        Rload = params.get('Rload', 5.0)
        
        v_avg = np.mean(Vout)
        v_ripple = np.max(Vout) - np.min(Vout)
        i_avg = np.mean(IL)
        i_max = np.max(IL)
        
        # 1. Open-circuit MOSFET
        # In a boost converter, if switch is open circuit, Vout stays equal to Vin (minus diode drop)
        # and current is near-zero under load.
        if v_avg < Vin * 1.05 and v_avg > Vin * 0.9 and i_avg < 0.2:
            faults.append("Open-circuit MOSFET Switch Fault detected")
            prob = max(prob, 0.95)
            
        # 2. Short-circuit MOSFET
        # Switch shorted -> Vin is shorted to ground through L. Huge current spike, Vout collapses to zero.
        if v_avg < 1.0 and i_max > (Vin / (params.get('DCR', 0.05) + 1e-9)):
            faults.append("Short-circuit MOSFET Switch Fault detected (Critical Over-current)")
            prob = max(prob, 0.99)
            
        # 3. Capacitor ESR degradation
        # Expected voltage ripple = I_ripple * ESR + D * I_out / (Fs * C)
        expected_ripple = (i_max - np.min(IL)) * ESR
        if v_ripple > 4.0 * expected_ripple and v_ripple > 1.5:
            faults.append("Capacitor ESR Degradation (Excessive Output Voltage Ripple)")
            prob = max(prob, 0.85)
            
        # 4. Inductor saturation
        # If peak current is extremely high, inductance drops, and slope dI/dt increases non-linearly
        if i_max > 30.0:
            faults.append("Inductor Core Saturation warning (High peak currents)")
            prob = max(prob, 0.75)
            
        # 5. Thermal Overload
        if temperatures.get('max_temp', 25.0) > 125.0:
            faults.append("Thermal Overload: Junction temperature exceeds 125°C limit")
            prob = max(prob, 0.90)

        # 6. Converter Instability
        # If output oscillates wildly
        if v_ripple / (v_avg + 1e-9) > 0.35:
            faults.append("PWM Loop / Converter Instability: Uncontrolled oscillations")
            prob = max(prob, 0.80)
            
        return {
            'faults_detected': faults,
            'fault_probability': prob,
            'status': 'Faulted' if len(faults) > 0 else 'Healthy'
        }
