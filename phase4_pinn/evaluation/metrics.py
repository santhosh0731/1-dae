"""
Evaluation Metrics — Physics + Prediction + Dynamic
======================================================
"""

import numpy as np
import torch
import json
from typing import Dict, Optional
from pathlib import Path


# ── Prediction Metrics ────────────────────────────────────────────────────────

def compute_prediction_metrics(
    y_true: np.ndarray,   # (N, 3) [Vout, IL, Vc]
    y_pred: np.ndarray,   # (N, 3)
    names:  list = ['Vout', 'IL', 'Vc']
) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    metrics = {}
    for i, name in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE']  = float(mean_absolute_error(t, p))
        metrics[f'{name}_RMSE'] = float(np.sqrt(mean_squared_error(t, p)))
        metrics[f'{name}_R2']   = float(r2_score(t, p))
    metrics['overall_R2'] = float(np.mean([metrics[f'{n}_R2'] for n in names]))
    return metrics


# ── Physics Metrics ───────────────────────────────────────────────────────────

def compute_physics_metrics(
    pred_np:   np.ndarray,  # (N, 5) [Vout, IL, Vc, dIL_dt, dVc_dt]
    params_np: np.ndarray,  # (N, 5) [Vin, D, L, C, Rload]
) -> Dict[str, float]:
    """Compute all physics residual norms."""
    Vout   = pred_np[:, 0]
    IL     = pred_np[:, 1]
    Vc     = pred_np[:, 2]
    dIL_dt = pred_np[:, 3]
    dVc_dt = pred_np[:, 4]

    Vin   = params_np[:, 0]
    D     = params_np[:, 1]
    L     = params_np[:, 2]
    C     = params_np[:, 3]
    Rload = params_np[:, 4]

    # KVL residual: L*dIL/dt - (Vin - (1-D)*Vout)
    r_kvl = L * dIL_dt - (Vin - (1.0 - D) * Vout)
    # KCL residual: C*dVc/dt - ((1-D)*IL - Vc/Rload)
    r_kcl = C * dVc_dt - ((1.0 - D) * IL - Vc / (Rload + 1e-9))
    # DAE algebraic: Vout - Vc
    r_dae = Vout - Vc
    # Power balance
    P_in  = Vin * IL * D
    P_out = Vout ** 2 / (Rload + 1e-9)
    r_pwr = P_in - P_out

    # Energy conservation error (%)
    e_conserv = np.abs(r_pwr) / (np.abs(P_in) + 1e-9) * 100.0

    return {
        'KVL_residual_norm':     float(np.sqrt(np.mean(r_kvl**2))),
        'KCL_residual_norm':     float(np.sqrt(np.mean(r_kcl**2))),
        'DAE_constraint_error':  float(np.sqrt(np.mean(r_dae**2))),
        'Power_balance_RMSE':    float(np.sqrt(np.mean(r_pwr**2))),
        'Energy_conserv_err_%':  float(np.mean(e_conserv)),
        'KVL_max':               float(np.max(np.abs(r_kvl))),
        'KCL_max':               float(np.max(np.abs(r_kcl))),
    }


# ── Dynamic Metrics ───────────────────────────────────────────────────────────

def compute_dynamic_metrics(
    y_true:     np.ndarray,   # (N, 3)
    y_pred:     np.ndarray,   # (N, 3)
    t:          np.ndarray,   # (N,) time axis
    Vout_ss:    Optional[float] = None,
    threshold:  float = 0.02,  # 2% settling criterion
) -> Dict[str, float]:
    """Waveform-level metrics."""
    Vout_true = y_true[:, 0]
    Vout_pred = y_pred[:, 0]
    IL_true   = y_true[:, 1]
    IL_pred   = y_pred[:, 1]

    metrics = {
        'Vout_ripple_true': float(np.max(Vout_true) - np.min(Vout_true)),
        'Vout_ripple_pred': float(np.max(Vout_pred) - np.min(Vout_pred)),
        'IL_peak_true':     float(np.max(IL_true)),
        'IL_peak_pred':     float(np.max(IL_pred)),
        'IL_peak_err_%':    float(abs(np.max(IL_pred) - np.max(IL_true)) /
                                  (np.max(IL_true) + 1e-9) * 100),
    }

    # Settling time error
    if Vout_ss is not None:
        def settling_time(Vout, t, ss, thr):
            for i in range(len(Vout)-1, -1, -1):
                if abs(Vout[i] - ss) / (ss + 1e-9) > thr:
                    return t[min(i+1, len(t)-1)]
            return 0.0
        ts_true = settling_time(Vout_true, t, Vout_ss, threshold)
        ts_pred = settling_time(Vout_pred, t, Vout_ss, threshold)
        metrics['settling_time_true_s'] = float(ts_true)
        metrics['settling_time_pred_s'] = float(ts_pred)
        metrics['settling_time_err_%']  = float(abs(ts_pred - ts_true) /
                                                  (ts_true + 1e-9) * 100)
    return metrics


# ── Full Evaluation ───────────────────────────────────────────────────────────

def evaluate_pinn(
    model,
    test_loader,
    scalers: Dict,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> Dict:
    """Complete PINN evaluation on test set."""
    model.eval()
    all_pred, all_true, all_params = [], [], []

    with torch.no_grad():
        for X, Y, P in test_loader:
            pred = model(X.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(Y.numpy())
            all_params.append(P.numpy())

    pred_norm  = np.vstack(all_pred)
    true_norm  = np.vstack(all_true)
    params_raw = np.vstack(all_params)

    # Inverse transform targets only (predictions are already physical)
    scaler_Y = scalers['Y']
    pred_real = pred_norm
    true_real = scaler_Y.inverse_transform(true_norm)[:, :3]

    # Metrics
    pred_metrics   = compute_prediction_metrics(true_real, pred_real[:, :3])
    physics_metrics = compute_physics_metrics(pred_real, params_raw)

    all_metrics = {**pred_metrics, **physics_metrics}

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    return all_metrics, pred_real, true_real, params_raw
