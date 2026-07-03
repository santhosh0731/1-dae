"""
run_phase3.py — Master Orchestration Script
============================================
Executes the complete Phase 3 DAE-PINN surrogate modeling pipeline:

  Stage 1: Data Validation
  Stage 2: Data Cleaning
  Stage 3: Feature Engineering
  Stage 4: Dataset Construction
  Level 1: Baseline Models (GPR, SVR, XGBoost, LightGBM, CatBoost, MLP)
  Level 2: Deep Learning Surrogates (CNN1D, TCN, Transformer, Autoencoder)
  Level 3: Operator Learning (DeepONet, FNO)
  Level 4: Continuous-Time Models (NeuralODE, LatentNeuralODE, ODE-RNN)
  Final:   Benchmarking & Best Model Selection

Usage:
  python run_phase3.py [--stage STAGE] [--level LEVEL]

  --stage : Run only up to this stage (1-4)
  --level : Run only this model level (1-4)
  --skip-training : Skip model training (benchmark only)
  --all   : Run everything (default)
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ─── Setup ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"phase3_run_{timestamp}.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "C:/Users/sanmu/Downloads/ltspice_raw.csv.csv"


# ─── Stage Functions ──────────────────────────────────────────────────────────

def header(title: str):
    logger.info("")
    logger.info("█" * 70)
    logger.info(f"  {title}")
    logger.info("█" * 70)


def run_data_pipeline():
    """Stages 1–4: Parse, validate, clean, engineer, construct datasets."""
    from src.pipeline.p01_data_validation   import run_validation
    from src.pipeline.p02_data_cleaning     import run_cleaning
    from src.pipeline.p03_feature_engineering import run_feature_engineering
    from src.pipeline.p04_dataset_construction import run_dataset_construction

    header("STAGE 1 — DATA VALIDATION")
    t0 = time.time()
    validation_result = run_validation(RAW_DATA_PATH)
    logger.info(f"  [DONE] Stage 1 complete in {time.time()-t0:.1f}s")
    logger.info(f"     Steps found: {validation_result.get('n_steps', 0)}")
    logger.info(f"     Issues: {validation_result.get('issues', [])}")

    header("STAGE 2 — DATA CLEANING")
    t0 = time.time()
    cleaned_result = run_cleaning(RAW_DATA_PATH, resample_n=512)
    logger.info(f"  [DONE] Stage 2 complete in {time.time()-t0:.1f}s")
    logger.info(f"     Clean steps: {cleaned_result['n_steps']}")
    logger.info(f"     Removed:     {cleaned_result['removed']}")

    header("STAGE 3 — FEATURE ENGINEERING")
    t0 = time.time()
    feature_result = run_feature_engineering(cleaned_result)
    logger.info(f"  [DONE] Stage 3 complete in {time.time()-t0:.1f}s")
    logger.info(f"     Scalar dataset shape: {feature_result['scalar_df'].shape}")

    header("STAGE 4 — DATASET CONSTRUCTION")
    t0 = time.time()
    dataset_result = run_dataset_construction(cleaned_result, feature_result)
    logger.info(f"  [DONE] Stage 4 complete in {time.time()-t0:.1f}s")

    return {
        'validation': validation_result,
        'cleaned': cleaned_result,
        'features': feature_result,
        'datasets': dataset_result
    }


def run_level1_training():
    header("LEVEL 1 — BASELINE SURROGATE MODELS")
    t0 = time.time()
    from src.models.level1_baselines.train_baselines import run_level1
    result = run_level1()
    logger.info(f"  [DONE] Level 1 complete in {time.time()-t0:.1f}s")
    return result


def run_level2_training():
    header("LEVEL 2 — DEEP LEARNING SURROGATES")
    t0 = time.time()
    from src.models.level2_deep_learning.train_deep_surrogates import run_level2
    result = run_level2()
    logger.info(f"  [DONE] Level 2 complete in {time.time()-t0:.1f}s")
    return result


def run_level3_training():
    header("LEVEL 3 — OPERATOR LEARNING MODELS")
    t0 = time.time()
    from src.models.level3_operator.train_operator_models import run_level3
    result = run_level3()
    logger.info(f"  [DONE] Level 3 complete in {time.time()-t0:.1f}s")
    return result


def run_level4_training():
    header("LEVEL 4 — CONTINUOUS-TIME DYNAMIC MODELS")
    t0 = time.time()
    from src.models.level4_continuous.train_continuous_models import run_level4
    result = run_level4()
    logger.info(f"  [DONE] Level 4 complete in {time.time()-t0:.1f}s")
    return result


def run_benchmarking_final():
    header("BENCHMARKING — BEST MODEL SELECTION")
    t0 = time.time()
    from src.evaluation.benchmarking import run_benchmarking
    result = run_benchmarking()
    logger.info(f"  [DONE] Benchmarking complete in {time.time()-t0:.1f}s")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 DAE-PINN Surrogate Training Pipeline")
    parser.add_argument('--stage', type=int, default=None,
                        help='Run only up to pipeline stage (1-4)')
    parser.add_argument('--level', type=int, default=None,
                        help='Run only this model level (1-4)')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data pipeline (assume datasets exist)')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Only run benchmarking summary')
    args = parser.parse_args()

    total_start = time.time()

    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PHASE 3 — ADVANCED SURROGATE MODELING FOR DAE-PINN FRAMEWORK  ║")
    logger.info("║  Foundation for Physics-Informed Neural Networks (PINNs)        ║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"  Run timestamp: {timestamp}")
    logger.info(f"  Dataset:       {RAW_DATA_PATH}")
    logger.info("")

    import torch
    device = "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
    logger.info(f"  Compute:       {device}")
    logger.info("")

    results = {}

    # ── Data Pipeline ──────────────────────────────────────────────────────
    if not args.skip_data and not args.benchmark_only:
        try:
            results['pipeline'] = run_data_pipeline()
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            logger.error("  -> Cannot continue without datasets")
            import traceback; traceback.print_exc()
            sys.exit(1)
    else:
        logger.info("  ⏭ Skipping data pipeline (--skip-data)")

    if args.benchmark_only:
        results['benchmark'] = run_benchmarking_final()
        return

    if args.stage is not None:
        logger.info(f"  Stopping after stage {args.stage} (--stage flag)")
        logger.info(f"  [DONE] Phase 3 Stages 1-{args.stage} Complete")
        return

    # ── Model Training ────────────────────────────────────────────────────
    level_runners = {
        1: run_level1_training,
        2: run_level2_training,
        3: run_level3_training,
        4: run_level4_training,
    }

    if args.level is not None:
        # Run only specified level
        if args.level in level_runners:
            try:
                results[f'level{args.level}'] = level_runners[args.level]()
            except Exception as e:
                logger.error(f"Level {args.level} failed: {e}")
                import traceback; traceback.print_exc()
    else:
        # Run all levels
        for lvl, runner in level_runners.items():
            try:
                results[f'level{lvl}'] = runner()
            except Exception as e:
                logger.error(f"Level {lvl} failed: {e}")
                import traceback; traceback.print_exc()
                logger.warning(f"  -> Continuing to next level...")

    # ── Final Benchmarking ────────────────────────────────────────────────
    try:
        results['benchmark'] = run_benchmarking_final()
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")

    total_time = time.time() - total_start

    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PHASE 3 COMPLETE                                               ║")
    logger.info(f"║  Total execution time: {total_time/60:.1f} minutes{' '*39}║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    logger.info("  📁 Outputs:")
    logger.info(f"     Results:    {BASE_DIR / 'results'}")
    logger.info(f"     Models:     {BASE_DIR / 'results' / 'models'}")
    logger.info(f"     Plots:      {BASE_DIR / 'results' / 'plots'}")
    logger.info(f"     Benchmarks: {BASE_DIR / 'results' / 'benchmarks'}")
    logger.info(f"     Logs:       {BASE_DIR / 'logs'}")

    if 'benchmark' in results and 'best' in results['benchmark']:
        logger.info("")
        logger.info("  [BEST] Best Models Selected:")
        for cat, info in results['benchmark']['best'].items():
            logger.info(f"     {cat.title():12}: {info['model']}  (R²={info.get('R2', 'N/A'):.4f})")


if __name__ == "__main__":
    main()
