"""
Prediction Metrics
===================
Calculates R2, MAE, and RMSE prediction metrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def compute_prediction_accuracy(
    y_true: np.ndarray,  # (N, 3) [Vout, IL, Vc]
    y_pred: np.ndarray,  # (N, 3)
) -> Dict[str, float]:
    """Return R2, MAE, RMSE metrics."""
    names = ['Vout', 'IL', 'Vc']
    metrics = {}
    for i, name in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mean_absolute_error(t, p))
        metrics[f'{name}_RMSE'] = float(np.sqrt(mean_squared_error(t, p)))
        metrics[f'{name}_R2'] = float(r2_score(t, p))

    metrics['overall_R2'] = float(np.mean([metrics[f'{n}_R2'] for n in names]))
    return metrics
