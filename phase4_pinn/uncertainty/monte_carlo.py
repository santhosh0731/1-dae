"""
Uncertainty Analysis
=====================
Monte Carlo Dropout inference and Sobol sensitivity analysis
for PINN predictions on boost converter.
"""

import numpy as np
import torch
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Monte Carlo Dropout ────────────────────────────────────────────────────────

class MCDropoutAnalyzer:
    """
    MC Dropout uncertainty estimation for PINN.

    Runs N stochastic forward passes with dropout enabled,
    computing mean prediction and epistemic uncertainty (std).
    """

    def __init__(self, model, device: torch.device, n_samples: int = 100):
        self.model     = model
        self.device    = device
        self.n_samples = n_samples

    def predict_with_uncertainty(
        self, X: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Run MC dropout and return mean + std per output.

        Returns:
            mean:  (B, 5) mean predictions
            std:   (B, 5) std (uncertainty)
            ci_lo: (B, 5) 5th percentile
            ci_hi: (B, 5) 95th percentile
        """
        X = X.to(self.device)
        mc = self.model.mc_predict(X, n_samples=self.n_samples)

        preds = mc['preds'].cpu().numpy()  # (N, B, 5)
        return {
            'mean':  preds.mean(axis=0),
            'std':   preds.std(axis=0),
            'ci_lo': np.percentile(preds, 5,  axis=0),
            'ci_hi': np.percentile(preds, 95, axis=0),
        }

    def summarize(self, uncertainty: Dict[str, np.ndarray]) -> Dict:
        """Aggregate uncertainty statistics over the test set."""
        output_names = ['Vout', 'IL', 'Vc', 'dIL_dt', 'dVc_dt']
        summary = {}
        for i, name in enumerate(output_names):
            summary[name] = {
                'mean_std':  float(uncertainty['std'][:, i].mean()),
                'max_std':   float(uncertainty['std'][:, i].max()),
                'ci_width':  float((uncertainty['ci_hi'][:, i] -
                                    uncertainty['ci_lo'][:, i]).mean()),
            }
        return summary


# ── Sobol Sensitivity Analysis ────────────────────────────────────────────────

class SobolSensitivityAnalyzer:
    """
    First-order Sobol sensitivity indices for PINN inputs.

    Measures: which input parameter [t, Vin, D, Fs, L, C, Rload]
    most affects Vout and IL predictions.

    Method: variance-based sensitivity (Monte Carlo estimation).
    """

    INPUT_NAMES = ['t', 'Vin', 'D', 'Fs', 'L', 'C', 'Rload']

    def __init__(self, model, scaler_X, device: torch.device, N: int = 2048):
        self.model    = model
        self.scaler_X = scaler_X
        self.device   = device
        self.N        = N  # MC samples for Sobol

    def _sample_inputs(self, n: int) -> np.ndarray:
        """Sample inputs from typical boost converter operating ranges."""
        rng = np.random.default_rng(42)
        raw = np.column_stack([
            rng.uniform(0, 0.005,   n),      # t   [s]
            rng.uniform(36, 60,     n),      # Vin [V]
            rng.uniform(0.4, 0.8,   n),      # D
            rng.choice([20e3, 50e3],n),      # Fs  [Hz]
            rng.uniform(50e-6,220e-6,n),     # L   [H]
            rng.uniform(47e-6,470e-6,n),     # C   [F]
            rng.uniform(1, 20,      n),      # Rload [Ω]
        ])
        return raw.astype(np.float32)

    def _predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Normalize and run PINN inference."""
        X_norm = self.scaler_X.transform(X_raw).astype(np.float32)
        X_t    = torch.tensor(X_norm).to(self.device)
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy()
        return pred

    def compute_first_order(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate first-order Sobol indices S_i for each input parameter.

        S_i = Var(E[Y|X_i]) / Var(Y)

        Approximated via pick-freeze method.
        """
        N = self.N
        A = self._sample_inputs(N)   # base sample
        B = self._sample_inputs(N)   # resample

        Y_A = self._predict(A)       # (N, 5)
        Y_B = self._predict(B)       # (N, 5)

        sobol = {}
        for i, name in enumerate(self.INPUT_NAMES):
            # Fix X_i in A to X_i from B → A_Bi
            A_Bi    = A.copy()
            A_Bi[:, i] = B[:, i]
            Y_ABi   = self._predict(A_Bi)

            # S_i = (Y_B * (Y_ABi - Y_A)).mean() / Var(Y_A)
            var_Y = np.var(Y_A, axis=0) + 1e-12   # (5,)
            cov_i = (Y_B * (Y_ABi - Y_A)).mean(axis=0)  # (5,)
            S_i   = cov_i / var_Y

            output_names = ['Vout', 'IL', 'Vc', 'dIL_dt', 'dVc_dt']
            sobol[name] = {
                out: float(np.clip(S_i[j], 0, 1))
                for j, out in enumerate(output_names)
            }

        return sobol

    def print_report(self, sobol: Dict) -> str:
        lines = ["\n  SOBOL SENSITIVITY INDICES (First-Order)\n"]
        header = f"  {'Input':<10}" + "".join(f"{'S_'+k:>12}" for k in ['Vout','IL','Vc'])
        lines.append(header)
        lines.append("  " + "-" * 50)
        for name, vals in sobol.items():
            row = f"  {name:<10}" + "".join(f"{vals[k]:>12.4f}" for k in ['Vout','IL','Vc'])
            lines.append(row)
        report = "\n".join(lines)
        logger.info(report)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger.info("Uncertainty analysis module loaded.")
    logger.info("Run via run_phase4.py --uncertainty")
