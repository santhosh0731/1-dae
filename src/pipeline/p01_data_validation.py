"""
Stage 1 — Data Validation
==========================
Validates the LTspice raw CSV before any processing.

Checks:
  - File accessibility and encoding
  - Step count vs. expected
  - Waveform continuity (monotonic time)
  - Duplicate step parameters
  - Missing / NaN values
  - Operating range sanity
  - Waveform length consistency
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.ltspice_parser import parse_ltspice_csv, build_params_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/01_validation.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical operating range limits (boost converter sanity)
# ---------------------------------------------------------------------------
LIMITS = {
    'Vin':   (10.0,   200.0),   # V
    'D':     (0.1,    0.95),    # duty cycle
    'Fs':    (1e3,    1e6),     # Hz
    'L':     (1e-9,   1e-1),    # H
    'C':     (1e-9,   1.0),     # F
    'Rload': (0.01,   1e4),     # Ω
}


def validate_file(filepath: str) -> bool:
    p = Path(filepath)
    if not p.exists():
        logger.error(f"File not found: {filepath}")
        return False
    if p.stat().st_size == 0:
        logger.error(f"File is empty: {filepath}")
        return False
    logger.info(f"[OK] File exists: {filepath}  ({p.stat().st_size / 1e6:.1f} MB)")
    return True


def validate_operating_ranges(params_df: pd.DataFrame) -> bool:
    logger.info("\n[Stage 1.4] Validating operating ranges...")
    all_ok = True
    for param, (lo, hi) in LIMITS.items():
        if param not in params_df.columns:
            logger.warning(f"  Parameter '{param}' not found in dataset")
            continue
        col = params_df[param].dropna()
        out_of_range = ((col < lo) | (col > hi)).sum()
        if out_of_range > 0:
            logger.warning(f"  [WARN] '{param}' has {out_of_range} out-of-range values "
                           f"(expected [{lo}, {hi}])")
            all_ok = False
        else:
            logger.info(f"  [OK] '{param}': range [{col.min():.4g}, {col.max():.4g}]  OK")
    return all_ok


def validate_waveforms(waveforms: list) -> bool:
    logger.info("\n[Stage 1.5] Validating waveform continuity...")
    all_ok = True
    nan_steps = []
    non_monotonic = []
    short_steps = []

    for i, df in enumerate(waveforms):
        # NaN check
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            nan_steps.append((i + 1, nan_count))

        # Monotonic time check
        time_diff = np.diff(df['time'].values)
        if (time_diff <= 0).any():
            non_monotonic.append(i + 1)

        # Minimum rows check (should have at least 100 points)
        if len(df) < 100:
            short_steps.append((i + 1, len(df)))

    if nan_steps:
        logger.warning(f"  [WARN] Steps with NaN values: {nan_steps}")
        all_ok = False
    else:
        logger.info(f"  [OK] No NaN values in any waveform")

    if non_monotonic:
        logger.warning(f"  [WARN] Non-monotonic time in steps: {non_monotonic}")
        all_ok = False
    else:
        logger.info(f"  [OK] All waveforms have monotonic time")

    if short_steps:
        logger.warning(f"  [WARN] Short waveform steps (<100 pts): {short_steps}")
        all_ok = False
    else:
        logger.info(f"  [OK] All steps have sufficient waveform points")

    return all_ok


def validate_duplicates(params_df: pd.DataFrame) -> bool:
    logger.info("\n[Stage 1.6] Checking for duplicate simulations...")
    param_cols = ['Vin', 'D', 'Fs', 'L', 'C', 'Rload']
    existing = [c for c in param_cols if c in params_df.columns]
    dup_count = params_df.duplicated(subset=existing).sum()
    if dup_count > 0:
        logger.warning(f"  [WARN] Found {dup_count} duplicate parameter combinations")
        return False
    logger.info(f"  [OK] No duplicate simulations found")
    return True


def run_validation(raw_path: str) -> dict:
    logger.info("=" * 70)
    logger.info("  STAGE 1 — DATA VALIDATION")
    logger.info("=" * 70)

    results = {'passed': True, 'issues': []}

    # Check 1: File accessibility
    if not validate_file(raw_path):
        results['passed'] = False
        results['issues'].append("File not found or empty")
        return results

    # Check 2: Parse the file
    logger.info("\n[Stage 1.2] Parsing LTspice raw CSV...")
    step_params, waveforms = parse_ltspice_csv(raw_path, verbose=True)

    if not step_params:
        logger.error("  [FAIL] No simulation steps found — file may be malformed")
        results['passed'] = False
        results['issues'].append("No simulation steps parsed")
        return results

    params_df = build_params_dataframe(step_params)

    logger.info(f"\n[Stage 1.3] Dataset overview:")
    logger.info(f"  Total steps parsed:    {len(step_params)}")
    logger.info(f"  Total waveform rows:   {sum(len(w) for w in waveforms):,}")
    logger.info(f"  Parameters per step:   {[c for c in params_df.columns if c not in ['step_num', 'step_total']]}")
    logger.info(f"\n  Parameter sweep:")
    for col in ['Vin', 'D', 'Fs', 'L', 'C', 'Rload']:
        if col in params_df.columns:
            uvals = sorted(params_df[col].dropna().unique().tolist())
            logger.info(f"    {col:8s}: {uvals}")

    # Check 3: Operating ranges
    if not validate_operating_ranges(params_df):
        results['issues'].append("Some parameters out of physical range")

    # Check 4: Waveform continuity
    if not validate_waveforms(waveforms):
        results['issues'].append("Waveform quality issues detected")

    # Check 5: Duplicates
    if not validate_duplicates(params_df):
        results['issues'].append("Duplicate simulations found")

    # Summary
    logger.info("\n" + "=" * 70)
    if not results['issues']:
        logger.info("  [DONE] VALIDATION PASSED — Dataset is clean and ready")
    else:
        logger.warning(f"  [WARN] VALIDATION COMPLETED WITH {len(results['issues'])} ISSUE(S):")
        for issue in results['issues']:
            logger.warning(f"      - {issue}")
        logger.info("  -> Proceeding to Stage 2 (Data Cleaning) to resolve issues")
    logger.info("=" * 70)

    results['n_steps'] = len(step_params)
    results['n_rows'] = sum(len(w) for w in waveforms)
    results['params_df'] = params_df
    results['waveforms'] = waveforms

    return results


if __name__ == "__main__":
    RAW_PATH = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"
    results = run_validation(RAW_PATH)
    print(f"\nValidation complete. Steps found: {results.get('n_steps', 0)}")
