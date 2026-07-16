"""
Adaptive Learning - Drift Detector
==================================
Identifies covariate shift or distribution drift on input telemetry parameters.
"""

import numpy as np

class InputDriftDetector:
    """Detects if new operating inputs deviate from trained ranges."""
    def __init__(self):
        # Nominal ranges for [Vin, D, Fs, L, C, Rload]
        self.nominal_ranges = {
            'Vin': (20.0, 100.0),
            'D': (0.3, 0.8),
            'Fs': (20000.0, 100000.0),
            'L': (20e-6, 200e-6),
            'C': (20e-6, 200e-6),
            'Rload': (2.0, 20.0)
        }

    def detect_drift(self, inputs: dict) -> dict:
        """Returns warnings for any parameter outside nominal ranges."""
        drift_signals = {}
        drift_detected = False
        
        for key, val in inputs.items():
            if key in self.nominal_ranges:
                lo, hi = self.nominal_ranges[key]
                if val < lo or val > hi:
                    drift_detected = True
                    drift_signals[key] = {
                        'status': 'Drift Warning',
                        'value': float(val),
                        'nominal_range': [lo, hi]
                    }
                else:
                    drift_signals[key] = {
                        'status': 'Normal',
                        'value': float(val)
                    }
                    
        return {
            'drift_detected': drift_detected,
            'report': drift_signals
        }
