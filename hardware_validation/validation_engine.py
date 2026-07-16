"""
Hardware Validation - Validation Engine
=======================================
Combines importer, metrics, and synchronizers. Kept inactive by default.
"""

from hardware_validation.experimental_importer import ExperimentalImporter
from hardware_validation.synchronization import TimeSynchronizer
from hardware_validation.comparison_metrics import ComparisonMetrics

class ValidationEngine:
    """Manages comparison of experimental measurements with solver/neural results."""
    def __init__(self):
        self.is_active = False # Disabled by default until physical setup is active
        
    def run_validation(self, csv_filepath: str, t_sim, y_sim):
        """Runs import, sync, and metric comparison if active."""
        if not self.is_active:
            return {
                'status': 'Inactive',
                'message': 'Experimental Hardware not available. Simulation validation only.'
            }
            
        df = ExperimentalImporter.load_csv(csv_filepath)
        if df is None:
            return {'status': 'Error', 'message': 'Failed to import CSV.'}
            
        t_exp = df['Time'].values
        v_exp = df['Vout'].values
        
        # Synchronize
        v_exp_sync = TimeSynchronizer.synchronize(t_sim, t_exp, v_exp)
        
        # Calculate comparison metrics
        metrics = ComparisonMetrics.compute_all(v_exp_sync, y_sim)
        return {
            'status': 'Active',
            'metrics': metrics
        }
