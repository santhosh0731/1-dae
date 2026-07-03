"""
Stage 2 — Data Cleaning
========================
Converts raw LTspice parsed data into a clean, structured dataset.

Tasks:
  - Remove corrupted / short waveform steps
  - Correct time inconsistencies
  - Convert engineering units to SI (already done in parser)
  - Organize waveform blocks by simulation step
  - Export cleaned parameter CSV + per-step waveform NPZ files
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.ltspice_parser import parse_ltspice_csv, build_params_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/02_cleaning.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS_PER_STEP = 100


def clean_waveform(df: pd.DataFrame, step_num: int) -> pd.DataFrame:
    """
    Clean a single waveform DataFrame:
      - Remove duplicate time stamps
      - Sort by time
      - Drop rows with NaN
      - Ensure monotonically increasing time
    """
    original_len = len(df)

    # Drop NaNs
    df = df.dropna()

    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)

    # Remove duplicate time stamps (keep first)
    df = df.drop_duplicates(subset='time', keep='first').reset_index(drop=True)

    # Ensure monotonic
    time_diff = np.diff(df['time'].values)
    if (time_diff <= 0).any():
        # Remove non-monotonic points
        mask = np.concatenate([[True], time_diff > 0])
        df = df[mask].reset_index(drop=True)

    cleaned_len = len(df)
    if cleaned_len < original_len:
        logger.debug(f"  Step {step_num}: removed {original_len - cleaned_len} points")

    return df


def resample_waveform(df: pd.DataFrame, n_points: int = 512) -> np.ndarray:
    """
    Resample waveform to uniform time grid with n_points.
    Returns array of shape (n_points, 8): [time, V_n001, ..., I_Vin]
    """
    t_min = df['time'].iloc[0]
    t_max = df['time'].iloc[-1]
    t_uniform = np.linspace(t_min, t_max, n_points)

    cols = ['time', 'V_n001', 'V_n002', 'V_n003', 'V_n004', 'I_L1', 'I_Rload', 'I_Vin']
    resampled = np.zeros((n_points, len(cols)))
    resampled[:, 0] = t_uniform

    for i, col in enumerate(cols[1:], start=1):
        try:
            f_interp = interp1d(df['time'].values, df[col].values,
                                kind='linear', bounds_error=False,
                                fill_value=(df[col].iloc[0], df[col].iloc[-1]))
            resampled[:, i] = f_interp(t_uniform)
        except Exception as e:
            logger.warning(f"Interpolation failed for {col}: {e}")

    return resampled


def run_cleaning(raw_path: str, resample_n: int = 512) -> dict:
    logger.info("=" * 70)
    logger.info("  STAGE 2 — DATA CLEANING")
    logger.info("=" * 70)

    logger.info("\n[Stage 2.1] Parsing raw data...")
    step_params, waveforms = parse_ltspice_csv(raw_path, verbose=True)
    params_df = build_params_dataframe(step_params)

    cleaned_waveforms = []
    cleaned_params = []
    removed_steps = []

    logger.info(f"\n[Stage 2.2] Cleaning {len(waveforms)} waveforms...")
    for i, (params, wf) in enumerate(zip(step_params, waveforms)):
        step_num = params.get('step_num', i + 1)
        wf_clean = clean_waveform(wf, step_num)

        if len(wf_clean) < MIN_ROWS_PER_STEP:
            logger.warning(f"  [FAIL] Step {step_num}: only {len(wf_clean)} rows after cleaning — REMOVED")
            removed_steps.append(step_num)
            continue

        cleaned_waveforms.append(wf_clean)
        cleaned_params.append(params)

    logger.info(f"\n  Kept:    {len(cleaned_waveforms)} steps")
    logger.info(f"  Removed: {len(removed_steps)} steps {removed_steps if removed_steps else '(none)'}")

    # Save cleaned params CSV
    clean_params_df = build_params_dataframe(cleaned_params)
    params_csv_path = PROCESSED_DIR / "params_clean.csv"
    clean_params_df.to_csv(params_csv_path, index=False)
    logger.info(f"\n[Stage 2.3] Saved cleaned parameters: {params_csv_path}")

    # Save raw waveforms as per-step NPZ
    logger.info(f"\n[Stage 2.4] Resampling waveforms to {resample_n} uniform points...")
    raw_npz_path = PROCESSED_DIR / "waveforms_raw.npz"
    resampled_npz_path = PROCESSED_DIR / "waveforms_resampled.npz"

    wf_data_raw = {}
    wf_data_resampled = {}

    for i, (params, wf) in enumerate(zip(cleaned_params, cleaned_waveforms)):
        step_num = int(params.get('step_num', i + 1))
        key = f"step_{step_num:03d}"

        # Raw waveform (variable length)
        wf_data_raw[key] = wf.values.astype(np.float32)

        # Resampled waveform (fixed length)
        resampled = resample_waveform(wf, n_points=resample_n)
        wf_data_resampled[key] = resampled.astype(np.float32)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(cleaned_waveforms)} waveforms...")

    np.savez_compressed(raw_npz_path, **wf_data_raw)
    np.savez_compressed(resampled_npz_path, **wf_data_resampled)
    logger.info(f"  [OK] Saved raw waveforms:      {raw_npz_path}")
    logger.info(f"  [OK] Saved resampled waveforms: {resampled_npz_path}")

    # Summary statistics
    logger.info(f"\n[Stage 2.5] Waveform statistics:")
    all_vout = np.array([wf['V_n003'].mean() for wf in cleaned_waveforms])
    all_il = np.array([wf['I_L1'].mean() for wf in cleaned_waveforms])
    logger.info(f"  Vout (avg per step): min={all_vout.min():.2f}V, max={all_vout.max():.2f}V")
    logger.info(f"  IL   (avg per step): min={all_il.min():.3f}A, max={all_il.max():.3f}A")

    logger.info("\n" + "=" * 70)
    logger.info("  [DONE] STAGE 2 COMPLETE — Data cleaning finished")
    logger.info("=" * 70)

    return {
        'n_steps': len(cleaned_waveforms),
        'removed': removed_steps,
        'params_df': clean_params_df,
        'waveforms': cleaned_waveforms,
        'wf_resampled_path': str(resampled_npz_path),
        'params_path': str(params_csv_path)
    }


if __name__ == "__main__":
    RAW_PATH = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"
    result = run_cleaning(RAW_PATH, resample_n=512)
    print(f"\nCleaning complete. {result['n_steps']} steps ready.")
