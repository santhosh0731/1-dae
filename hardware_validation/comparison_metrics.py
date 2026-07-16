"""
Hardware Validation - Comparison Metrics
========================================
Error metric calculations comparing experimental vs. simulation states.
"""

import numpy as np

class ComparisonMetrics:
    """Computes precision validation scores."""
    
    @staticmethod
    def compute_all(y_true, y_pred):
        """Computes RMSE, MAE, MAPE, R2, and Cross Correlation."""
        y_true = np.array(y_true, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=np.float32)
        
        # Match lengths
        min_len = min(len(y_true), len(y_pred))
        y_t = y_true[:min_len]
        y_p = y_pred[:min_len]
        
        # 1. RMSE
        rmse = np.sqrt(np.mean((y_t - y_p)**2))
        
        # 2. MAE
        mae = np.mean(np.abs(y_t - y_p))
        
        # 3. MAPE
        mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-9))) * 100.0
        
        # 4. R2
        ss_res = np.sum((y_t - y_p)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
        
        # 5. Cross Correlation
        corr = 0.0
        if len(y_t) > 1 and np.std(y_t) > 1e-6 and np.std(y_p) > 1e-6:
            corr = float(np.corrcoef(y_t, y_p)[0, 1])
            
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'cross_correlation': corr
        }
