"""
Intelligent Diagnostics - Anomaly Detector
===========================================
Monitors residual mismatches between physical solver state and neural surrogate outputs.
"""

import numpy as np

class AnomalyDetector:
    """Detects signal and physics anomalies in real-time prediction curves."""
    
    @staticmethod
    def detect_anomalies(y_solver, y_ai):
        """Compares physical vs. AI prediction arrays. Flags errors above threshold."""
        if isinstance(y_ai, str) or y_ai is None:
            return {'anomaly_detected': False, 'residual_mae': 0.0}
            
        # Match shapes
        min_len = min(len(y_solver), len(y_ai))
        mae = np.mean(np.abs(y_solver[:min_len, 0] - y_ai[:min_len, 0]))
        
        # Define threshold of 5.0 V deviation
        anomaly = bool(mae > 5.0)
        
        return {
            'anomaly_detected': anomaly,
            'residual_mae': float(mae),
            'severity': 'High' if mae > 10.0 else ('Medium' if anomaly else 'Low')
        }
