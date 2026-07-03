"""
Stage 4 — Dataset Construction
================================
Builds three research-ready datasets from cleaned + engineered features:

1. Scalar Dataset    — input params -> steady-state scalar targets
2. Waveform Dataset  — input params -> full Vout(t), IL(t) waveforms
3. Dynamic Dataset   — (params, t) -> state [IL(t), Vout(t)] per time point

Also applies normalization and train/val/test splits.
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.normalizer import DataNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/04_dataset_construction.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SCALAR_DIR  = BASE_DIR / "data" / "scalar_dataset"
WAVEFORM_DIR = BASE_DIR / "data" / "waveform_dataset"
DYNAMIC_DIR  = BASE_DIR / "data" / "dynamic_dataset"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
for d in [WAVEFORM_DIR, DYNAMIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

INPUT_FEATURES = ['Vin', 'D', 'Fs', 'L', 'C', 'Rload']
SCALAR_TARGETS = ['Vout_avg', 'Vout_ripple', 'Voltage_gain', 'IL_avg',
                  'IL_peak', 'IL_rms', 'Iout_avg', 'Power_out',
                  'Efficiency', 'Settling_time']
WAVEFORM_SIGNALS = ['V_n003', 'I_L1']   # Vout, IL


def split_indices(n: int, seed: int = RANDOM_SEED):
    """Return train, val, test index arrays."""
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=(VAL_RATIO + TEST_RATIO),
                                           random_state=seed, shuffle=True)
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size),
                                         random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx


def build_scalar_dataset(scalar_df: pd.DataFrame) -> dict:
    """Build normalized scalar dataset with splits."""
    logger.info("\n[Stage 4.1] Building Scalar Dataset...")

    # Filter valid rows
    df = scalar_df.copy()
    valid_cols = INPUT_FEATURES + SCALAR_TARGETS
    df = df[valid_cols].dropna().reset_index(drop=True)
    logger.info(f"  Valid rows: {len(df)}")

    X = df[INPUT_FEATURES].values.astype(np.float32)
    Y = df[SCALAR_TARGETS].values.astype(np.float32)

    train_idx, val_idx, test_idx = split_indices(len(df))

    # Fit scalers on training data only
    X_scaler = DataNormalizer(method='standard', feature_names=INPUT_FEATURES)
    Y_scaler = DataNormalizer(method='standard', feature_names=SCALAR_TARGETS)

    X_train = X_scaler.fit_transform(X[train_idx])
    X_val   = X_scaler.transform(X[val_idx])
    X_test  = X_scaler.transform(X[test_idx])

    Y_train = Y_scaler.fit_transform(Y[train_idx])
    Y_val   = Y_scaler.transform(Y[val_idx])
    Y_test  = Y_scaler.transform(Y[test_idx])

    # Save
    np.savez_compressed(SCALAR_DIR / "scalar_train.npz", X=X_train, Y=Y_train)
    np.savez_compressed(SCALAR_DIR / "scalar_val.npz",   X=X_val,   Y=Y_val)
    np.savez_compressed(SCALAR_DIR / "scalar_test.npz",  X=X_test,  Y=Y_test)

    # Save raw (unscaled) for inverse transform during evaluation
    np.savez_compressed(SCALAR_DIR / "scalar_raw.npz",
                        X=X, Y=Y,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    X_scaler.save(SCALAR_DIR / "X_scaler.joblib")
    Y_scaler.save(SCALAR_DIR / "Y_scaler.joblib")

    # Save metadata
    meta = {
        'n_total': int(len(df)),
        'n_train': int(len(train_idx)),
        'n_val':   int(len(val_idx)),
        'n_test':  int(len(test_idx)),
        'input_features': INPUT_FEATURES,
        'target_features': SCALAR_TARGETS,
        'X_scaler': X_scaler.summary(),
        'Y_scaler': Y_scaler.summary(),
    }
    with open(SCALAR_DIR / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  [OK] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    logger.info(f"  [OK] Input shape: {X.shape}, Target shape: {Y.shape}")
    logger.info(f"  [OK] Saved to: {SCALAR_DIR}")

    return {'X_scaler': X_scaler, 'Y_scaler': Y_scaler,
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
            'X_raw': X, 'Y_raw': Y}


def build_waveform_dataset(params_df: pd.DataFrame, waveforms: list,
                           n_points: int = 512) -> dict:
    """Build waveform dataset: params -> [Vout(t), IL(t)] arrays."""
    logger.info("\n[Stage 4.2] Building Waveform Dataset...")
    from scipy.interpolate import interp1d

    param_cols = [c for c in INPUT_FEATURES if c in params_df.columns]
    X_list, Y_list, T_list = [], [], []
    valid_indices = []

    for i, (_, row) in enumerate(params_df.iterrows()):
        wf = waveforms[i]
        if len(wf) < 50:
            continue

        x = np.array([row[c] for c in param_cols], dtype=np.float32)

        # Resample to uniform grid
        t_min, t_max = wf['time'].iloc[0], wf['time'].iloc[-1]
        t_uniform = np.linspace(t_min, t_max, n_points)

        try:
            vout_interp = interp1d(wf['time'].values, wf['V_n003'].values,
                                   kind='linear', fill_value='extrapolate')
            il_interp   = interp1d(wf['time'].values, wf['I_L1'].values,
                                   kind='linear', fill_value='extrapolate')
            vout_u = vout_interp(t_uniform).astype(np.float32)
            il_u   = il_interp(t_uniform).astype(np.float32)
        except Exception as e:
            logger.warning(f"  Step {i}: interpolation failed: {e}")
            continue

        y = np.stack([vout_u, il_u], axis=-1)  # (T, 2)
        t_norm = (t_uniform - t_uniform[0]) / (t_uniform[-1] - t_uniform[0] + 1e-12)

        X_list.append(x)
        Y_list.append(y)
        T_list.append(t_norm.astype(np.float32))
        valid_indices.append(i)

    X = np.array(X_list, dtype=np.float32)   # (N, 6)
    Y = np.array(Y_list, dtype=np.float32)   # (N, T, 2)
    T = np.array(T_list, dtype=np.float32)   # (N, T)

    logger.info(f"  X shape: {X.shape}, Y shape: {Y.shape}, T shape: {T.shape}")

    n = len(X)
    train_idx, val_idx, test_idx = split_indices(n)

    # Normalize X
    X_scaler = DataNormalizer(method='standard', feature_names=param_cols)
    X_norm = X_scaler.fit_transform(X)

    # Normalize Y per-channel (across all steps and time)
    Y_flat = Y.reshape(-1, 2)
    Y_scaler = DataNormalizer(method='standard', feature_names=['Vout', 'IL'])
    Y_flat_norm = Y_scaler.fit_transform(Y_flat)
    Y_norm = Y_flat_norm.reshape(Y.shape)

    # Save splits
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        np.savez_compressed(WAVEFORM_DIR / f"waveform_{split_name}.npz",
                            X=X_norm[idx], Y=Y_norm[idx], T=T[idx],
                            X_raw=X[idx], Y_raw=Y[idx])

    X_scaler.save(WAVEFORM_DIR / "X_scaler.joblib")
    Y_scaler.save(WAVEFORM_DIR / "Y_scaler.joblib")

    meta = {
        'n_total': n, 'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
        'n_time_points': n_points, 'input_features': param_cols,
        'output_signals': ['Vout', 'IL'],
        'X_shape': list(X.shape), 'Y_shape': list(Y.shape),
    }
    with open(WAVEFORM_DIR / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  [OK] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    logger.info(f"  [OK] Saved to: {WAVEFORM_DIR}")

    return {'X_scaler': X_scaler, 'Y_scaler': Y_scaler,
            'X': X_norm, 'Y': Y_norm, 'T': T,
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}


def build_dynamic_dataset(params_df: pd.DataFrame, waveforms: list) -> dict:
    """
    Build dynamic dataset: at each time point -> state [IL, Vout].
    Used for Neural ODE and DAE training.
    Each sample = (params, time_array, state_trajectory).
    """
    logger.info("\n[Stage 4.3] Building Dynamic Dataset...")
    from scipy.interpolate import interp1d

    param_cols = [c for c in INPUT_FEATURES if c in params_df.columns]
    samples = []

    for i, (_, row) in enumerate(params_df.iterrows()):
        wf = waveforms[i]
        if len(wf) < 50:
            continue

        params = np.array([row[c] for c in param_cols], dtype=np.float32)
        t = wf['time'].values.astype(np.float32)
        IL   = wf['I_L1'].values.astype(np.float32)
        Vout = wf['V_n003'].values.astype(np.float32)

        # State = [IL, Vout]
        state = np.stack([IL, Vout], axis=-1)  # (T, 2)

        samples.append({
            'params': params,
            'time':   t,
            'state':  state,
            'step_num': int(row.get('step_num', i)),
        })

    logger.info(f"  Total dynamic samples: {len(samples)}")

    n = len(samples)
    train_idx, val_idx, test_idx = split_indices(n)

    # Save as NPZ (variable-length time -> list of arrays)
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_samples = [samples[i] for i in idx]
        save_dict = {}
        for j, s in enumerate(split_samples):
            save_dict[f"params_{j}"] = s['params']
            save_dict[f"time_{j}"]   = s['time']
            save_dict[f"state_{j}"]  = s['state']
        np.savez_compressed(DYNAMIC_DIR / f"dynamic_{split_name}.npz", **save_dict)
        logger.info(f"  [OK] {split_name}: {len(split_samples)} trajectories")

    meta = {
        'n_total': n, 'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
        'state_dim': 2, 'state_names': ['IL', 'Vout'],
        'input_features': param_cols,
        'n_per_split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}
    }
    with open(DYNAMIC_DIR / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  [OK] Saved to: {DYNAMIC_DIR}")
    return {'samples': samples, 'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}


def run_dataset_construction(cleaned_result: dict, feature_result: dict) -> dict:
    logger.info("=" * 70)
    logger.info("  STAGE 4 — DATASET CONSTRUCTION")
    logger.info("=" * 70)

    scalar_df  = feature_result['scalar_df']
    params_df  = cleaned_result['params_df']
    waveforms  = cleaned_result['waveforms']

    scalar_out  = build_scalar_dataset(scalar_df)
    waveform_out = build_waveform_dataset(params_df, waveforms, n_points=512)
    dynamic_out  = build_dynamic_dataset(params_df, waveforms)

    logger.info("\n" + "=" * 70)
    logger.info("  [DONE] STAGE 4 COMPLETE — All datasets constructed and saved")
    logger.info(f"  Scalar    -> {SCALAR_DIR}")
    logger.info(f"  Waveform  -> {WAVEFORM_DIR}")
    logger.info(f"  Dynamic   -> {DYNAMIC_DIR}")
    logger.info("=" * 70)

    return {'scalar': scalar_out, 'waveform': waveform_out, 'dynamic': dynamic_out}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.pipeline.p02_data_cleaning import run_cleaning
    from src.pipeline.p03_feature_engineering import run_feature_engineering

    RAW_PATH = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"
    cleaned  = run_cleaning(RAW_PATH)
    features = run_feature_engineering(cleaned)
    result   = run_dataset_construction(cleaned, features)
    print("Dataset construction complete!")
