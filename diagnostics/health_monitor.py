"""
Intelligent Diagnostics - Health Monitor
=========================================
Aggregates fault probabilities, temperature margins, and residual errors into a unified Health Index.
"""

class HealthMonitor:
    """Consolidates system health metrics (0% to 100%)."""
    
    @staticmethod
    def compute_health_index(fault_results, anomaly_results, temperatures):
        """Calculates a percentage score representing converter health."""
        score = 100.0
        
        # Deduct for faults
        prob = fault_results.get('fault_probability', 0.0)
        score -= prob * 70.0
        
        # Deduct for anomalies
        if anomaly_results.get('anomaly_detected', False):
            score -= 15.0
            
        # Deduct for thermal margins
        max_t = temperatures.get('max_temp', 25.0)
        if max_t > 90.0:
            overheat_factor = min(1.0, (max_t - 90.0) / 35.0) # scaling from 90 to 125 C
            score -= overheat_factor * 15.0
            
        final_score = max(0.0, min(100.0, score))
        return {
            'health_index': final_score,
            'health_class': 'Healthy' if final_score > 85.0 else ('Degraded' if final_score > 50.0 else 'Critical'),
            'thermal_margin_c': float(max(0.0, 125.0 - max_t))
        }
