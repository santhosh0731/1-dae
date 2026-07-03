"""
Stage 3 — Feature Engineering
==============================
Computes all derived features from cleaned waveforms:

Input Features:
  Vin, D, Fs, L, C, Rload

Dynamic Features:
  time, Vsw (V_n001), Vin_node (V_n002), Vout (V_n003),
  Vgate (V_n004), IL (I_L1), Iout (I_Rload), Iin (I_Vin)

Scalar Output Targets (per step):
  Vout_avg, Vout_ripple, IL_avg, IL_peak, IL_rms, IL_ripple,
  Iout_avg, Iin_avg, Power_in, Power_out, Efficiency, Voltage_gain,
  Settling_time

Physics Features:
  KVL_residual, KCL_residual, dIL_dt, dVout_dt
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/03_feature_engineering.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SCALAR_DIR = BASE_DIR / "data" / "scalar_dataset"
PHYSICS_DIR = BASE_DIR / "data" / "physics_dataset"
for d in [SCALAR_DIR, PHYSICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Steady-state fraction: use the last STEADY_FRAC of the waveform
STEADY_FRAC = 0.3


def compute_scalar_features(params: dict, wf: pd.DataFrame) -> dict:
    """Compute all scalar steady-state features for one step."""
    # Use last STEADY_FRAC of waveform for steady-state averaging
    n = len(wf)
    ss_start = int(n * (1 - STEADY_FRAC))
    wf_ss = wf.iloc[ss_start:].copy()

    Vin  = params.get('Vin', float('nan'))
    D    = params.get('D', float('nan'))
    Fs   = params.get('Fs', float('nan'))
    L    = params.get('L', 50e-6)
    C    = params.get('C', 47e-6)
    Rload = params.get('Rload', 1.0)

    # --- Voltage ---
    Vout_avg    = wf_ss['V_n003'].mean()
    Vout_max    = wf_ss['V_n003'].max()
    Vout_min    = wf_ss['V_n003'].min()
    Vout_ripple = Vout_max - Vout_min
    Voltage_gain = Vout_avg / Vin if Vin != 0 else float('nan')

    # Theoretical boost gain: 1 / (1 - D)
    Vout_theoretical = Vin / (1 - D) if D < 1 else float('nan')
    Gain_error = abs(Vout_avg - Vout_theoretical) / Vout_theoretical * 100 if np.isfinite(Vout_theoretical) else float('nan')

    # --- Inductor Current ---
    IL_avg    = wf_ss['I_L1'].mean()
    IL_max    = wf_ss['I_L1'].max()
    IL_min    = wf_ss['I_L1'].min()
    IL_peak   = IL_max
    IL_ripple = IL_max - IL_min
    IL_rms    = np.sqrt(np.mean(wf_ss['I_L1'].values ** 2))

    # --- Output / Input ---
    Iout_avg = wf_ss['I_Rload'].mean()
    Iin_avg  = abs(wf_ss['I_Vin'].mean())

    # --- Power ---
    Power_out = Vout_avg * abs(Iout_avg)
    Power_in  = Vin * Iin_avg if Iin_avg > 0 else float('nan')
    Efficiency = (Power_out / Power_in * 100) if (np.isfinite(Power_in) and Power_in > 0) else float('nan')

    # --- Settling Time ---
    # Time to reach within 2% of final Vout_avg
    Vout_full = wf['V_n003'].values
    time_full = wf['time'].values
    target = Vout_avg
    tol = 0.02 * target if target > 0 else 0.01
    settled_mask = np.abs(Vout_full - target) < tol
    settled_idx = np.argmax(settled_mask) if settled_mask.any() else len(time_full) - 1
    Settling_time = time_full[settled_idx]

    return {
        'Vin': Vin, 'D': D, 'Fs': Fs, 'L': L, 'C': C, 'Rload': Rload,
        'Vout_avg': Vout_avg,
        'Vout_ripple': Vout_ripple,
        'Vout_theoretical': Vout_theoretical,
        'Gain_error_pct': Gain_error,
        'Voltage_gain': Voltage_gain,
        'IL_avg': IL_avg,
        'IL_peak': IL_peak,
        'IL_ripple': IL_ripple,
        'IL_rms': IL_rms,
        'Iout_avg': Iout_avg,
        'Iin_avg': Iin_avg,
        'Power_in': Power_in,
        'Power_out': Power_out,
        'Efficiency': Efficiency,
        'Settling_time': Settling_time,
        'step_num': params.get('step_num', -1),
    }


def compute_physics_features(params: dict, wf: pd.DataFrame) -> pd.DataFrame:
    """Compute point-wise physics residual features."""
    L     = params.get('L', 50e-6)
    C     = params.get('C', 47e-6)
    Rload = params.get('Rload', 1.0)
    Vin   = params.get('Vin', 36.0)

    time  = wf['time'].values
    IL    = wf['I_L1'].values
    Vout  = wf['V_n003'].values
    Vsw   = wf['V_n001'].values
    Iout  = wf['I_Rload'].values

    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1] if len(dt) > 1 else 1e-9  # avoid zero

    # Numerical derivatives
    dIL_dt   = np.gradient(IL, time)
    dVout_dt = np.gradient(Vout, time)

    # KVL: Vin - L*(dIL/dt) - Vsw ≈ 0  (when switch ON)
    KVL_residual = Vin - L * dIL_dt - Vsw

    # KCL: IL - Iout - C*(dVout/dt) ≈ 0
    KCL_residual = IL - Iout - C * dVout_dt

    # Inductor dynamics: V_L = L * dIL/dt
    V_L = L * dIL_dt
    V_C = Vout

    phys_df = pd.DataFrame({
        'time': time,
        'IL': IL,
        'Vout': Vout,
        'Vsw': Vsw,
        'Iout': Iout,
        'dIL_dt': dIL_dt,
        'dVout_dt': dVout_dt,
        'KVL_residual': KVL_residual,
        'KCL_residual': KCL_residual,
        'V_L': V_L,
        'V_C': V_C,
    })

    # Broadcast step params
    for k, v in [('Vin', Vin), ('D', params.get('D', 0.5)),
                 ('Fs', params.get('Fs', 20e3)), ('L', L), ('C', C), ('Rload', Rload)]:
        phys_df[k] = v

    phys_df['step_num'] = int(params.get('step_num', -1))
    return phys_df


def run_feature_engineering(cleaned_result: dict) -> dict:
    logger.info("=" * 70)
    logger.info("  STAGE 3 — FEATURE ENGINEERING")
    logger.info("=" * 70)

    step_params_list = []
    # Re-read params
    params_df = cleaned_result['params_df']
    waveforms  = cleaned_result['waveforms']

    # Rebuild step_params from params_df rows
    param_records = params_df.to_dict(orient='records')

    # --- Scalar Features ---
    logger.info("\n[Stage 3.1] Computing scalar steady-state features...")
    scalar_rows = []
    physics_dfs = []

    for i, (params, wf) in enumerate(zip(param_records, waveforms)):
        scalar = compute_scalar_features(params, wf)
        scalar_rows.append(scalar)

        # Physics features (subsampled for efficiency)
        stride = max(1, len(wf) // 512)
        wf_sub = wf.iloc[::stride].reset_index(drop=True)
        phys = compute_physics_features(params, wf_sub)
        physics_dfs.append(phys)

        if (i + 1) % 20 == 0:
            logger.info(f"  Computed features for {i+1}/{len(waveforms)} steps...")

    scalar_df = pd.DataFrame(scalar_rows)
    logger.info(f"\n  [OK] Scalar dataset shape: {scalar_df.shape}")

    # Save scalar dataset
    scalar_path = SCALAR_DIR / "scalar_features.csv"
    scalar_df.to_csv(scalar_path, index=False)
    logger.info(f"  [OK] Saved scalar features: {scalar_path}")

    # --- Physics Features ---
    logger.info("\n[Stage 3.2] Saving physics feature dataset...")
    physics_df = pd.concat(physics_dfs, ignore_index=True)
    physics_path = PHYSICS_DIR / "physics_features.parquet"
    physics_df.to_parquet(physics_path, index=False)
    logger.info(f"  [OK] Physics dataset shape: {physics_df.shape}")
    logger.info(f"  [OK] Saved physics features: {physics_path}")

    # --- Summary ---
    logger.info(f"\n[Stage 3.3] Scalar feature summary:")
    target_cols = ['Vout_avg', 'Efficiency', 'Voltage_gain', 'Vout_ripple',
                   'IL_peak', 'IL_rms', 'Settling_time']
    for col in target_cols:
        if col in scalar_df.columns:
            logger.info(f"  {col:20s}: min={scalar_df[col].min():.4g}, "
                        f"max={scalar_df[col].max():.4g}, "
                        f"mean={scalar_df[col].mean():.4g}")

    logger.info("\n" + "=" * 70)
    logger.info("  [DONE] STAGE 3 COMPLETE — Feature engineering finished")
    logger.info("=" * 70)

    return {
        'scalar_df': scalar_df,
        'scalar_path': str(scalar_path),
        'physics_path': str(physics_path),
        'physics_df': physics_df
    }


if __name__ == "__main__":
    # Run after Stage 2
    from src.pipeline._02_data_cleaning import run_cleaning
    RAW_PATH = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"
    cleaned = run_cleaning(RAW_PATH)
    result = run_feature_engineering(cleaned)
    print(f"\nFeature engineering complete. Scalar shape: {result['scalar_df'].shape}")
