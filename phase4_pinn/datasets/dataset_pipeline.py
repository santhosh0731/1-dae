"""
Phase 4 Dataset Pipeline
=========================
Loads Phase 3 physics data, applies physics feature engineering,
constructs boundary/initial conditions, and normalizes for PINN training.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Dict
import pickle

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)

PHYS_PARQUET = BASE_DIR / "data" / "physics_dataset" / "physics_features.parquet"
OUT_DIR      = BASE_DIR / "phase4_pinn" / "datasets"

# Input features fed to PINN
INPUT_COLS  = ['time', 'Vin', 'D', 'Fs', 'L', 'C', 'Rload']
# Target outputs [Vout, IL, Vc] — Vc = Vout for boost converter
TARGET_COLS = ['Vout', 'IL', 'Vout']   # Vc = Vout (capacitor = output)


class PhysicsDatasetBuilder:
    """Build the PINN physics dataset from Phase 3 output."""

    def __init__(self, parquet_path: Path = PHYS_PARQUET):
        self.parquet_path = parquet_path
        self.df: pd.DataFrame = None
        self.scalers: Dict = {}

    def load(self) -> 'PhysicsDatasetBuilder':
        logger.info(f"Loading physics dataset: {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        logger.info(f"  Shape: {self.df.shape}")
        logger.info(f"  Columns: {list(self.df.columns)}")
        return self

    def engineer_physics_features(self) -> 'PhysicsDatasetBuilder':
        """Add physics feature engineering (Improvement #2)."""
        df = self.df
        L = df['L'].values
        C = df['C'].values
        IL = df['IL'].values
        Vout = df['Vout'].values
        Vin = df['Vin'].values
        D = df['D'].values
        Rload = df['Rload'].values

        df['E_L']            = 0.5 * L * IL**2
        df['E_C']            = 0.5 * C * Vout**2
        df['P_in']           = Vin * IL * D + 1e-9
        df['P_out']          = Vout**2 / (Rload + 1e-9)
        df['E_conserv_err']  = np.abs(df['P_in'] - df['P_out']) / df['P_in'] * 100.0
        df['efficiency']     = np.clip(df['P_out'] / df['P_in'] * 100.0, 0, 100)
        df['duty_region']    = (D >= 0.5).astype(float)
        # CCM flag: IL_min > 0 (approximate: IL always positive if IL > 0)
        df['CCM_flag']       = (IL > 0).astype(float)

        logger.info(f"  Physics features added: E_L, E_C, P_in, P_out, efficiency, ...")
        self.df = df
        return self

    def build_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract initial (t=0) and boundary (t=T) conditions per step."""
        ic_list, bc_list = [], []
        for step_id, grp in self.df.groupby('step_num'):
            grp_sorted = grp.sort_values('time')
            t0_row = grp_sorted.iloc[0]
            tT_row = grp_sorted.iloc[-1]

            params = grp_sorted[['Vin','D','L','C','Rload']].iloc[0].values

            # IC: [t=0, Vin, D, L, C, Rload, IL0, Vout0, Vc0]
            ic_list.append(np.concatenate([
                [0.0], params,
                [t0_row['IL'], t0_row['Vout'], t0_row['Vout']]
            ]))
            # BC: [t=T, Vin, D, L, C, Rload, IL_ss, Vout_ss, Vc_ss]
            bc_list.append(np.concatenate([
                [tT_row['time']], params,
                [tT_row['IL'], tT_row['Vout'], tT_row['Vout']]
            ]))

        ic = np.array(ic_list)
        bc = np.array(bc_list)
        logger.info(f"  IC shape: {ic.shape}, BC shape: {bc.shape}")

        np.save(OUT_DIR / "initial_conditions" / "ic.npy", ic)
        np.save(OUT_DIR / "boundary_conditions" / "bc.npy", bc)
        return ic, bc

    def normalize(self) -> 'PhysicsDatasetBuilder':
        """Normalize inputs and targets. Save scalers for inference."""
        from sklearn.preprocessing import StandardScaler

        X = self.df[INPUT_COLS].values.astype(np.float32)
        Y_Vout = self.df['Vout'].values.astype(np.float32)
        Y_IL   = self.df['IL'].values.astype(np.float32)
        # Vc = Vout for ideal boost converter
        Y_Vc   = Y_Vout.copy()
        # Derivatives
        Y_dIL  = self.df['dIL_dt'].values.astype(np.float32)
        Y_dVc  = self.df['dVout_dt'].values.astype(np.float32)

        Y = np.column_stack([Y_Vout, Y_IL, Y_Vc, Y_dIL, Y_dVc])  # (N, 5)

        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_norm = scaler_X.fit_transform(X).astype(np.float32)
        Y_norm = scaler_Y.fit_transform(Y).astype(np.float32)

        self.scalers = {'X': scaler_X, 'Y': scaler_Y}

        # Stage 8: Compute reference scales from raw data
        self.scalers['physics_scales'] = {
            'V_REF': float(np.mean(np.abs(Y_Vout))),
            'I_REF': float(np.mean(np.abs(Y_IL))),
            'P_REF': float(np.mean(np.abs(self.df['Vin'].values * Y_IL)))
        }

        # Save
        norm_dir = OUT_DIR / "normalized"
        norm_dir.mkdir(parents=True, exist_ok=True)
        np.save(norm_dir / "X_norm.npy", X_norm)
        np.save(norm_dir / "Y_norm.npy", Y_norm)
        with open(norm_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)

        # Also save physics params for loss [Vin, D, L, C, Rload]
        params_cols = ['Vin', 'D', 'L', 'C', 'Rload']
        params = self.df[params_cols].values.astype(np.float32)
        np.save(norm_dir / "params.npy", params)

        # Also save Fs for reference (not used in physics but in inputs)
        np.save(norm_dir / "Fs.npy", self.df['Fs'].values.astype(np.float32))

        logger.info(f"  X_norm: {X_norm.shape}, Y_norm: {Y_norm.shape}")
        logger.info(f"  Scalers saved: {norm_dir / 'scalers.pkl'}")
        return self

    def build(self) -> Dict:
        """Full pipeline."""
        (OUT_DIR / "initial_conditions").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "boundary_conditions").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "normalized").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "physics_features_engineered").mkdir(parents=True, exist_ok=True)

        self.load()
        self.engineer_physics_features()

        # Save engineered features
        eng_path = OUT_DIR / "physics_features_engineered" / "features.parquet"
        self.df.to_parquet(eng_path)
        logger.info(f"  Engineered features saved: {eng_path}")

        ic, bc = self.build_conditions()
        self.normalize()

        logger.info("  [OK] Dataset pipeline complete.")
        return {
            'n_samples': len(self.df),
            'n_steps': self.df['step_num'].nunique(),
            'scalers': self.scalers,
            'ic': ic,
            'bc': bc,
        }


class PINNDataset(Dataset):
    """PyTorch Dataset for PINN training."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, params: np.ndarray):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.Y      = torch.tensor(Y, dtype=torch.float32)
        self.params = torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.params[idx]


def load_pinn_dataloaders(
    batch_size: int = 512,
    train_split: float = 0.70,
    val_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Load normalized data and return train/val/test DataLoaders."""
    norm_dir = OUT_DIR / "normalized"

    X      = np.load(norm_dir / "X_norm.npy")
    Y      = np.load(norm_dir / "Y_norm.npy")
    params = np.load(norm_dir / "params.npy")

    with open(norm_dir / "scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)

    dataset = PINNDataset(X, Y, params)
    N = len(dataset)
    n_train = int(N * train_split)
    n_val   = int(N * val_split)
    n_test  = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader, scalers


def build_dataset():
    """Entry point for dataset construction."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger.info("=" * 60)
    logger.info("  PHASE 4 — DATASET PIPELINE")
    logger.info("=" * 60)
    result = PhysicsDatasetBuilder().build()
    logger.info(f"  Samples: {result['n_samples']:,}, Steps: {result['n_steps']}")
    return result


if __name__ == "__main__":
    build_dataset()
