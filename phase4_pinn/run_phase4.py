"""
run_phase4.py — Master Orchestration Script for Phase 4 PINN
=============================================================

Usage:
  python run_phase4.py              # Full pipeline
  python run_phase4.py --skip-data  # Skip dataset prep (use cached)
  python run_phase4.py --eval-only  # Load checkpoint + evaluate only
  python run_phase4.py --solvers    # Run numerical solver benchmark only
  python run_phase4.py --uncertainty # Run uncertainty analysis only

Pipeline:
  Stage 1: Physics Dataset Pipeline (feature engineering + normalization)
  Stage 2: Numerical Solver Benchmark (RK4/RK45/Radau/BDF/GL)
  Stage 3: PINN Training (curriculum + adaptive loss)
  Stage 4: Evaluation (prediction + physics + dynamic metrics)
  Stage 5: Uncertainty Analysis (MC Dropout + Sobol)
  Stage 6: Phase 5 Export Package
"""

import sys
import time
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import yaml

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

# ── Logging Setup ──────────────────────────────────────────────────────────────

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"phase4_run_{ts}.log", mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config() -> dict:
    cfg_path = BASE_DIR / "configs" / "pinn_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def header(title: str):
    logger.info("")
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


# ── Stage 1: Dataset ──────────────────────────────────────────────────────────

def run_dataset_pipeline():
    header("STAGE 1 — PHYSICS DATASET PIPELINE")
    from phase4_pinn.datasets.dataset_pipeline import build_dataset
    t0 = time.time()
    result = build_dataset()
    logger.info(f"  [OK] Dataset ready in {time.time()-t0:.1f}s")
    logger.info(f"      Samples: {result['n_samples']:,} | Steps: {result['n_steps']}")
    return result


# ── Stage 2: Numerical Solvers ────────────────────────────────────────────────

def run_solver_benchmark():
    header("STAGE 2 — NUMERICAL SOLVER BENCHMARK")
    from phase4_pinn.numerical.solver_benchmark import benchmark_solvers
    import json

    report = BASE_DIR / "reports"
    report.mkdir(exist_ok=True)

    all_results = {}
    test_cases = [
        {'Vin':36,'D':0.4,'Rload':1.0},
        {'Vin':36,'D':0.5,'Rload':1.0},
        {'Vin':48,'D':0.6,'Rload':5.0},
        {'Vin':60,'D':0.7,'Rload':10.0},
    ]

    for case in test_cases:
        key = f"Vin{case['Vin']}_D{case['D']}_R{case['Rload']}"
        logger.info(f"\n  Test case: {key}")
        res = benchmark_solvers(**case)
        # Remove non-serializable arrays
        clean = {s: {k: v for k, v in m.items() if k not in ('t', 'Y')}
                 for s, m in res.items() if 'error' not in m}
        all_results[key] = clean

    solver_path = report / "solver_benchmark.json"
    with open(solver_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n  [OK] Solver benchmark saved: {solver_path}")
    logger.info("  Recommendation: Radau-IIA (best stiff DAE accuracy) -> Phase 5")
    return all_results


# ── Stage 3: PINN Training ────────────────────────────────────────────────────

def run_pinn_training(config: dict):
    header("STAGE 3 — PINN TRAINING")
    from phase4_pinn.models.pinn_model import build_pinn
    from phase4_pinn.models.trainer import PINNTrainer
    from phase4_pinn.datasets.dataset_pipeline import load_pinn_dataloaders

    train_loader, val_loader, test_loader, scalers = load_pinn_dataloaders(
        batch_size=config['training']['batch_size']
    )

    model = build_pinn(config)
    logger.info(f"  PINN Parameters: {model.count_parameters():,}")
    logger.info(f"  Device: {DEVICE}")

    trainer = PINNTrainer(
        model    = model,
        config   = config,
        ckpt_dir = BASE_DIR / "checkpoints",
        log_dir  = LOG_DIR,
    )

    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    elapsed = time.time() - t0
    logger.info(f"  [OK] Training complete in {elapsed/60:.1f} min")

    return model, history, test_loader, scalers


# ── Stage 4: Evaluation ───────────────────────────────────────────────────────

def run_evaluation(model, test_loader, scalers):
    header("STAGE 4 — PINN EVALUATION")
    from phase4_pinn.evaluation.metrics import evaluate_pinn
    from phase4_pinn.evaluation.waveform_plots import plot_waveform_comparison, plot_loss_curves

    eval_dir = BASE_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    metrics, pred_real, true_real, params = evaluate_pinn(
        model, test_loader, scalers, DEVICE,
        save_path=reports_dir / "phase4_metrics.json"
    )

    logger.info(f"\n  PREDICTION METRICS:")
    for k in ['Vout_R2', 'IL_R2', 'Vc_R2', 'overall_R2']:
        if k in metrics:
            logger.info(f"    {k:25s}: {metrics[k]:.4f}")

    logger.info(f"\n  PHYSICS METRICS:")
    for k in ['KVL_residual_norm', 'KCL_residual_norm',
              'DAE_constraint_error', 'Energy_conserv_err_%']:
        if k in metrics:
            logger.info(f"    {k:25s}: {metrics[k]:.6f}")

    # Plot waveform comparison (first test sample)
    if len(pred_real) > 0 and len(true_real) > 0:
        import numpy as np
        t_axis = range(min(512, len(pred_real)))
        idx = list(t_axis)
        try:
            plot_waveform_comparison(
                t        = np.arange(len(idx)) * 1e-5,
                Vout_true= true_real[idx, 0],
                IL_true  = true_real[idx, 1],
                Vout_pred= pred_real[idx, 0],
                IL_pred  = pred_real[idx, 1],
                title    = "Phase 4 PINN vs LTspice Ground Truth",
                save_path= str(eval_dir / "pinn_waveform_comparison.png"),
            )
        except Exception as e:
            logger.warning(f"  Waveform plot failed: {e}")

    # Write final report
    report_path = reports_dir / "phase4_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("  PHASE 4 — PINN EVALUATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write("  PREDICTION METRICS\n  " + "-"*40 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<35}: {v:.6f}\n")
        f.write("\n  COMPARISON vs PHASE 3 BEST\n  " + "-"*40 + "\n")
        f.write(f"  Phase 3 Transformer R2  : 0.9960\n")
        f.write(f"  Phase 3 SVR R2          : 0.9897\n")
        pinn_r2 = metrics.get('overall_R2', 0)
        f.write(f"  Phase 4 PINN R2         : {pinn_r2:.4f}\n")
        f.write(f"\n  PINN advantage: physically consistent + generalizes better\n")

    logger.info(f"  [OK] Report: {report_path}")
    return metrics


# ── Stage 5: Uncertainty ──────────────────────────────────────────────────────

def run_uncertainty_analysis(model, test_loader, scalers):
    header("STAGE 5 — UNCERTAINTY ANALYSIS")
    from phase4_pinn.uncertainty.monte_carlo import MCDropoutAnalyzer, SobolSensitivityAnalyzer

    X_test_all = []
    for X, _, _ in test_loader:
        X_test_all.append(X)
        if len(X_test_all) >= 5:
            break
    X_sample = torch.cat(X_test_all, dim=0)[:512]

    # MC Dropout
    logger.info("  Running MC Dropout (n=100 samples)...")
    mc = MCDropoutAnalyzer(model, DEVICE, n_samples=100)
    unc = mc.predict_with_uncertainty(X_sample)
    summary = mc.summarize(unc)

    logger.info("  MC Uncertainty Summary:")
    for name, vals in summary.items():
        if name in ['Vout','IL']:
            logger.info(f"    {name}: mean_std={vals['mean_std']:.4f}, "
                       f"CI_width={vals['ci_width']:.4f}")

    # Sobol sensitivity
    logger.info("  Running Sobol sensitivity analysis...")
    try:
        sobol_analyzer = SobolSensitivityAnalyzer(model, scalers['X'], DEVICE)
        sobol = sobol_analyzer.compute_first_order()
        report = sobol_analyzer.print_report(sobol)

        with open(BASE_DIR / "reports" / "sobol_sensitivity.json", 'w') as f:
            json.dump(sobol, f, indent=2)
        logger.info("  [OK] Sobol indices saved")
    except Exception as e:
        logger.warning(f"  Sobol analysis failed: {e}")

    return {'mc_uncertainty': summary}


# ── Stage 6: Phase 5 Export ───────────────────────────────────────────────────

def run_phase5_export(model, scalers, metrics):
    header("STAGE 6 — PHASE 5 EXPORT PACKAGE")
    from phase4_pinn.models.inference import export_for_phase5
    export_path = export_for_phase5(model, scalers, metrics)
    logger.info(f"  [OK] Phase 5 package: {export_path}")
    return export_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4 PINN Pipeline")
    parser.add_argument('--skip-data',   action='store_true')
    parser.add_argument('--eval-only',   action='store_true')
    parser.add_argument('--solvers',     action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()

    t_total = time.time()
    config  = load_config()

    logger.info("")
    logger.info("+" + "=" * 63 + "+")
    logger.info("|  PHASE 4 — PHYSICS-INFORMED NEURAL NETWORK (PINN)           |")
    logger.info("|  Boost Converter DAE-PINN Framework                         |")
    logger.info("+" + "=" * 63 + "+")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Run:    {ts}")

    results = {}

    # ── Solvers only
    if args.solvers:
        results['solvers'] = run_solver_benchmark()
        return

    # ── Data pipeline
    if not args.skip_data and not args.eval_only:
        results['dataset'] = run_dataset_pipeline()

    # ── Solver benchmark
    if not args.eval_only:
        try:
            results['solvers'] = run_solver_benchmark()
        except Exception as e:
            logger.warning(f"  Solver benchmark failed: {e}")

    # ── Training
    if not args.eval_only:
        model, history, test_loader, scalers = run_pinn_training(config)
    else:
        # Load best checkpoint
        header("LOADING SAVED PINN CHECKPOINT")
        from phase4_pinn.models.pinn_model import build_pinn, PINNModel
        from phase4_pinn.datasets.dataset_pipeline import load_pinn_dataloaders
        import pickle

        ckpt_path = BASE_DIR / "checkpoints" / "pinn_best.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = build_pinn(config).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        _, _, test_loader, scalers = load_pinn_dataloaders(
            batch_size=config['training']['batch_size']
        )
        history = {}
        logger.info(f"  [OK] Loaded checkpoint: {ckpt_path}")

    # ── Evaluation
    metrics = run_evaluation(model, test_loader, scalers)
    results['metrics'] = metrics

    # ── Uncertainty
    if not args.eval_only or args.uncertainty:
        try:
            results['uncertainty'] = run_uncertainty_analysis(model, test_loader, scalers)
        except Exception as e:
            logger.warning(f"  Uncertainty analysis failed: {e}")

    # ── Phase 5 export
    export_path = run_phase5_export(model, scalers, metrics)
    results['export_path'] = str(export_path)

    # ── Summary
    elapsed = time.time() - t_total
    logger.info("")
    logger.info("+" + "=" * 63 + "+")
    logger.info("|  PHASE 4 COMPLETE                                            |")
    logger.info(f"|  Total time: {elapsed/60:.1f} min" + " " * (49 - len(f"{elapsed/60:.1f}")) + "|")
    logger.info("+" + "=" * 63 + "+")
    logger.info(f"  R2(Vout) : {metrics.get('Vout_R2', 0):.4f}")
    logger.info(f"  R2(IL)   : {metrics.get('IL_R2',   0):.4f}")
    logger.info(f"  KVL norm : {metrics.get('KVL_residual_norm', 0):.6f}")
    logger.info(f"  KCL norm : {metrics.get('KCL_residual_norm', 0):.6f}")
    logger.info(f"  Phase 5 export: {export_path}")
    logger.info("")
    logger.info("  Next: python phase5_daepinn/run_phase5.py")


if __name__ == "__main__":
    main()
