"""
Evaluation Metrics
==================
All metrics used to benchmark surrogate models.
"""

import time
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from typing import Dict, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    prefix: str = "") -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray  shape (N,) or (N, M)
    y_pred : np.ndarray  shape (N,) or (N, M)
    prefix : str         Optional metric name prefix

    Returns
    -------
    dict with MAE, MSE, RMSE, R2, MAPE
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE (avoid division by zero)
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float('nan')

    p = prefix + "_" if prefix else ""
    return {
        f"{p}MAE": mae,
        f"{p}MSE": mse,
        f"{p}RMSE": rmse,
        f"{p}R2": r2,
        f"{p}MAPE": mape,
    }


def compute_waveform_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              signal_names: Optional[list] = None) -> Dict[str, float]:
    """
    Compute per-signal waveform reconstruction metrics.

    Parameters
    ----------
    y_true : np.ndarray  (N, T, S) — N samples, T time points, S signals
    y_pred : np.ndarray  (N, T, S)
    signal_names : list of str, optional
    """
    if y_true.ndim == 2:
        y_true = y_true[:, :, None]
        y_pred = y_pred[:, :, None]

    n_signals = y_true.shape[-1]
    if signal_names is None:
        signal_names = [f"sig{i}" for i in range(n_signals)]

    all_metrics = {}
    for s, name in enumerate(signal_names):
        yt = y_true[:, :, s].flatten()
        yp = y_pred[:, :, s].flatten()
        m = compute_metrics(yt, yp, prefix=name)
        all_metrics.update(m)

    # Overall
    yt_all = y_true.flatten()
    yp_all = y_pred.flatten()
    all_metrics.update(compute_metrics(yt_all, yp_all, prefix="overall"))

    return all_metrics


class ModelTimer:
    """Context manager for timing training / inference."""
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.label}: {self.elapsed:.3f}s"
