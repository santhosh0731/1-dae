"""
run_phase5.py — Master Orchestration Script for Phase 5 DAE-PINN
==================================================================

Orchestrates the final DAE-PINN pipeline with embedded Radau-IIA IRK integrations,
NTK gradient balancing, evaluation suites, and ONNX deployment export.
"""

import sys
import os
import time
import logging
import argparse
import pickle
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    os.environ["PYTHONUTF8"] = "1"

from phase5_dae_pinn.models.dae_pinn import DAEPINNModel
from phase5_dae_pinn.models.trainer import DAEPINNTrainer
from phase5_dae_pinn.models.inference import load_dae_pinn_model
from phase4_pinn.datasets.dataset_pipeline import load_pinn_dataloaders
from phase5_dae_pinn.irk.radau_iia import solve_radau_iia
from phase5_dae_pinn.dae.dae_formulation import evaluate_dae_numpy
from phase5_dae_pinn.evaluation.prediction_metrics import compute_prediction_accuracy
from phase5_dae_pinn.evaluation.physics_metrics import compute_physics_violations
from phase5_dae_pinn.evaluation.irk_metrics import compute_irk_integration_metrics
from phase5_dae_pinn.evaluation.comparison import generate_comparison_table
from phase5_dae_pinn.deployment.export_model import export_to_onnx_and_script

# Logging setup
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"phase5_run_{ts}.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config() -> dict:
    with open(BASE_DIR / "configs" / "phase5_config.yaml") as f:
        return yaml.safe_load(f)


def run_solver_benchmarks():
    logger.info("\n" + "="*60 + "\n  STAGE 2: IMPLICIT RUNGE-KUTTA SOLVER COMPARISONS\n" + "="*60)
    # Define test parameters
    Vin, D, L, C, Rload = 36.0, 0.5, 50e-6, 47e-6, 1.0
    y0 = np.array([0.01, 5.0])
    t_span = (0.0, 0.005)
    t_eval = np.linspace(0.0, 0.005, 512)

    def boost_ode(t, y, Vin, D, L, C, Rload):
        IL, Vc = y
        dIL_dt = (Vin - (1.0 - D) * Vc) / L
        dVc_dt = ((1.0 - D) * IL - Vc / Rload) / C
        return np.array([dIL_dt, dVc_dt])

    # Benchmark Radau-IIA (High-Accuracy implicit stiff DAE solver)
    t_sol, Y_sol = solve_radau_iia(boost_ode, t_span, y0, t_eval, Vin=Vin, D=D, L=L, C=C, Rload=Rload)
    logger.info(f"  Radau-IIA integration complete. State-space final state: IL={Y_sol[-1, 0]:.4f} A, Vc={Y_sol[-1, 1]:.4f} V")
    return t_sol, Y_sol


def run_dae_pinn_training(config: dict):
    logger.info("\n" + "="*60 + "\n  STAGE 3: DAE-PINN TRAINING WITH EMBEDDED IRK\n" + "="*60)
    train_loader, val_loader, test_loader, scalers = load_pinn_dataloaders(
        batch_size=config['training']['batch_size']
    )

    model = DAEPINNModel(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model'].get('dropout_rate', 0.05),
    )

    trainer = DAEPINNTrainer(
        model=model,
        config=config,
        ckpt_dir=BASE_DIR / "checkpoints",
        log_dir=LOG_DIR,
        scalers=scalers,
    )

    t0 = time.time()
    trainer.fit(train_loader, val_loader)
    logger.info(f"  [OK] Training completed in {(time.time() - t0)/60:.2f} mins")

    return model, test_loader, scalers


def run_evaluation(model, test_loader, scalers, config: dict):
    logger.info("\n" + "="*60 + "\n  STAGE 4: SYSTEM CONVERGENCE & ACCURACY EVALUATION\n" + "="*60)
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Gather test predictions
    model.eval()
    all_pred, all_true, all_params = [], [], []
    with torch.no_grad():
        for X, Y, P in test_loader:
            pred = model(X.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(Y.numpy())
            all_params.append(P.numpy())

    pred_norm = np.vstack(all_pred)
    true_norm = np.vstack(all_true)
    params_raw = np.vstack(all_params)

    # Inverse normalize targets only (predictions are already physical)
    scaler_Y = scalers['Y']
    pred_real = pred_norm
    true_real = scaler_Y.inverse_transform(true_norm)[:, :3]

    # Evaluate prediction metrics
    pred_metrics = compute_prediction_accuracy(true_real, pred_real[:, :3])
    logger.info("\n  Prediction Accuracy Results:")
    for k, v in pred_metrics.items():
        logger.info(f"    {k:<20}: {v:.5f}")

    # Evaluate physics law consistency
    phys_metrics = compute_physics_violations(pred_real, params_raw)
    logger.info("\n  Physics law violations (Residual L2 norms):")
    for k, v in phys_metrics.items():
        logger.info(f"    {k:<25}: {v:.6f}")

    # Evaluate numerical embedded IRK residuals
    t_std = float(scalers['X'].scale_[0])
    irk_metrics = compute_irk_integration_metrics(model, test_loader, config, device, t_std)
    logger.info("\n  Embedded Implicit Runge-Kutta Step Deviation:")
    for k, v in irk_metrics.items():
        logger.info(f"    {k:<25}: {v:.6f}")

    # Export configuration and summary metrics
    summary = {**pred_metrics, **phys_metrics, **irk_metrics}
    metrics_file = reports_dir / "phase5_metrics.json"
    with open(metrics_file, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    logger.info(f"\n  [OK] Exported raw json metrics: {metrics_file.name}")

    # Generate final consolidated report
    generate_comparison_table(summary, reports_dir)

    # ONNX and TorchScript exports
    export_to_onnx_and_script(model, BASE_DIR / "deployment")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5 DAE-PINN Engine")
    parser.add_argument('--eval-only', action='store_true', help="Skip train, evaluate from checkpoint")
    args = parser.parse_args()

    config = load_config()

    logger.info("+" + "="*63 + "+")
    logger.info("|  PHASE 5 — DAE-PINN SYSTEM INTEGRATION COMPLETE RUN          |")
    logger.info("|  Scientific Machine Learning (SciML) Boost Converter Model  |")
    logger.info("+" + "="*63 + "+")

    # Stage 2 Solver comparisons
    run_solver_benchmarks()

    if not args.eval_only:
        # Stage 3 Training
        model, test_loader, scalers = run_dae_pinn_training(config)
    else:
        # Load best trained model checkpoint
        ckpt_path = BASE_DIR / "checkpoints" / "dae_pinn_best.pt"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}. Please train the model first.")
            return
        model, _ = load_dae_pinn_model(ckpt_path, device)
        _, _, test_loader, scalers = load_pinn_dataloaders(config['training']['batch_size'])
        logger.info(f"Loaded checkpoint successfully: {ckpt_path.name}")

    # Stage 4 Evaluation
    run_evaluation(model, test_loader, scalers, config)


if __name__ == "__main__":
    main()
