"""
LTspice Raw CSV Parser
======================
Parses LTspice .raw exported CSV files with:
  - "Step Information: Vin=36 D=400m Fs=20K L=50µ C=47µ Rload=1" headers
  - Tab-separated waveform data rows
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Engineering unit multipliers (LTspice notation includes µ as byte)
# ------------------------------------------------------------------
_UNIT_MAP = {
    'T': 1e12, 'G': 1e9, 'Meg': 1e6, 'K': 1e3, 'k': 1e3,
    'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15,
}
# µ appears as various byte sequences in latin-1 encoded files
_MU_BYTES = ['\xb5', '\xb5', 'µ', 'u']  # latin-1 µ, unicode µ, ascii u fallback


def _parse_value(val_str: str) -> float:
    """Convert LTspice value string to SI float (e.g. '400m' -> 0.4, '50µ' -> 50e-6)."""
    val_str = val_str.strip()
    # Replace µ variants with 'u'
    for mu in _MU_BYTES:
        val_str = val_str.replace(mu, 'u')
    # Try suffix match
    for suffix, multiplier in sorted(_UNIT_MAP.items(), key=lambda x: -len(x[0])):
        if val_str.endswith(suffix):
            try:
                return float(val_str[:-len(suffix)]) * multiplier
            except ValueError:
                pass
    # No suffix — plain number
    try:
        return float(val_str)
    except ValueError:
        logger.warning(f"Cannot parse value: '{val_str}', defaulting to NaN")
        return float('nan')


def _parse_step_info(line: str) -> Optional[Dict[str, float]]:
    """
    Parse a Step Information line.

    Example:
        'Step Information: Vin=36 D=400m Fs=20K L=50µ C=47µ Rload=1  (Step: 1/94)'

    Returns dict like:
        {'Vin': 36.0, 'D': 0.4, 'Fs': 20000.0, 'L': 50e-6, 'C': 47e-6, 'Rload': 1.0,
         'step_num': 1, 'step_total': 94}
    """
    if not line.startswith("Step Information:"):
        return None

    params = {}

    # Extract step number
    step_match = re.search(r'\(Step:\s*(\d+)/(\d+)\)', line)
    if step_match:
        params['step_num'] = int(step_match.group(1))
        params['step_total'] = int(step_match.group(2))

    # Extract key=value pairs — value may contain unit suffix including µ-variants
    kv_pattern = r'(\w+)=([^\s\(]+)'
    for key, val in re.findall(kv_pattern, line):
        if key not in ('Step',):  # skip keyword fragments
            params[key] = _parse_value(val)

    return params


def parse_ltspice_csv(
    filepath: str,
    encoding: str = 'latin-1',
    verbose: bool = True
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    Parse LTspice raw CSV export.

    Parameters
    ----------
    filepath : str
        Path to the LTspice raw CSV file.
    encoding : str
        File encoding (default: 'latin-1').
    verbose : bool
        Print progress.

    Returns
    -------
    step_params : list of dicts
        List of parameter dicts for each simulation step.
    waveforms : list of DataFrames
        List of waveform DataFrames, one per step, with columns:
        ['time', 'V_n001', 'V_n002', 'V_n003', 'V_n004', 'I_L1', 'I_Rload', 'I_Vin']
    """
    WAVEFORM_COLS = ['time', 'V_n001', 'V_n002', 'V_n003', 'V_n004',
                     'I_L1', 'I_Rload', 'I_Vin']

    step_params: List[Dict] = []
    waveforms: List[pd.DataFrame] = []

    current_params: Optional[Dict] = None
    current_rows: List[List[float]] = []

    filepath = Path(filepath)
    total_size = filepath.stat().st_size
    processed = 0

    if verbose:
        print(f"[Parser] Reading: {filepath}")
        print(f"[Parser] File size: {total_size / 1e6:.1f} MB")

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        # Skip header line
        header = f.readline()

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip('\r\n')
            processed += len(line) + 1

            if not line.strip():
                continue

            if line.startswith("Step Information:"):
                # Save previous step
                if current_params is not None and current_rows:
                    df = pd.DataFrame(current_rows, columns=WAVEFORM_COLS, dtype=np.float64)
                    waveforms.append(df)
                    step_params.append(current_params)

                current_params = _parse_step_info(line)
                current_rows = []

                if verbose and len(step_params) % 10 == 0:
                    pct = 100 * processed / total_size
                    print(f"  [Parser] Steps parsed: {len(step_params)} | Progress: {pct:.1f}%")
            else:
                # Data row — tab-separated
                parts = line.split('\t')
                if len(parts) == 8:
                    try:
                        row = [float(v) for v in parts]
                        current_rows.append(row)
                    except ValueError:
                        logger.debug(f"Line {line_num}: skipping malformed row: {line[:60]}")

    # Save last step
    if current_params is not None and current_rows:
        df = pd.DataFrame(current_rows, columns=WAVEFORM_COLS, dtype=np.float64)
        waveforms.append(df)
        step_params.append(current_params)

    if verbose:
        print(f"\n[Parser] [OK] Parsed {len(step_params)} simulation steps")
        if step_params:
            rows_per_step = [len(w) for w in waveforms]
            print(f"[Parser]   Rows per step — min: {min(rows_per_step)}, "
                  f"max: {max(rows_per_step)}, avg: {np.mean(rows_per_step):.0f}")

    return step_params, waveforms


def build_params_dataframe(step_params: List[Dict]) -> pd.DataFrame:
    """Convert list of step parameter dicts to a clean DataFrame."""
    df = pd.DataFrame(step_params)
    # Ensure standard column ordering
    priority_cols = ['step_num', 'step_total', 'Vin', 'D', 'Fs', 'L', 'C', 'Rload']
    cols = [c for c in priority_cols if c in df.columns]
    rest = [c for c in df.columns if c not in priority_cols]
    df = df[cols + rest].reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    raw_path = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"
    params, waves = parse_ltspice_csv(raw_path, verbose=True)
    df_params = build_params_dataframe(params)
    print("\nParameter sweep summary:")
    print(df_params.describe())
    print("\nFirst step waveform shape:", waves[0].shape)
    print(waves[0].head(3))
