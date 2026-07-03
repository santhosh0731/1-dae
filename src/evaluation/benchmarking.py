"""
Model Benchmarking
==================
Aggregates all benchmark results across Levels 1-4 and produces
a final comparison report with the best model selection.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parents[2]
BENCH_DIR = BASE_DIR / "results" / "benchmarks"
PLOTS_DIR = BASE_DIR / "results" / "plots"


def load_all_benchmarks() -> Dict:
    """Load all saved benchmark JSON files."""
    all_results = {}
    for level_file in sorted(BENCH_DIR.glob("level*_benchmark.json")):
        level = level_file.stem.split('_')[0]
        try:
            with open(level_file) as f:
                data = json.load(f)
            all_results[level] = data
            logger.info(f"  Loaded: {level_file.name} ({len(data)} models)")
        except Exception as e:
            logger.warning(f"  Could not load {level_file}: {e}")
    return all_results


def extract_key_metrics(all_results: Dict) -> pd.DataFrame:
    """Build a flat comparison DataFrame across all models."""
    rows = []
    for level, models in all_results.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                row = {'Level': level, 'Model': model_name}
                # Extract primary metrics with flexible key search
                for target_key in ['Vout_avg_R2', 'overall_R2', 'Vout_R2', 'IL_R2']:
                    if target_key in metrics:
                        row['R2_primary'] = metrics[target_key]
                        row['R2_key'] = target_key
                        break
                for rmse_key in ['Vout_avg_RMSE', 'overall_RMSE', 'Vout_RMSE']:
                    if rmse_key in metrics:
                        row['RMSE_primary'] = metrics[rmse_key]
                        break
                row['Train_time_s']      = metrics.get('train_time_s', float('nan'))
                row['Inference_time_ms'] = metrics.get('inference_time_ms', float('nan'))
                row['N_params']          = metrics.get('n_params', float('nan'))
                rows.append(row)
    df = pd.DataFrame(rows)
    if 'R2_primary' in df.columns:
        df = df.sort_values('R2_primary', ascending=False)
    return df


def select_best_models(comparison_df: pd.DataFrame) -> Dict:
    """Select best model per category."""
    best = {}

    # Best scalar surrogate (Level 1)
    l1 = comparison_df[comparison_df['Level'] == 'level1']
    if not l1.empty and 'R2_primary' in l1.columns:
        best_scalar = l1.loc[l1['R2_primary'].idxmax()]
        best['scalar'] = {'model': best_scalar['Model'],
                          'R2': best_scalar['R2_primary'],
                          'level': 'Level 1'}

    # Best waveform surrogate (Level 2 + 3)
    l23 = comparison_df[comparison_df['Level'].isin(['level2', 'level3'])]
    if not l23.empty and 'R2_primary' in l23.columns:
        best_wf = l23.loc[l23['R2_primary'].idxmax()]
        best['waveform'] = {'model': best_wf['Model'],
                            'R2': best_wf['R2_primary'],
                            'level': best_wf['Level']}

    # Best dynamic surrogate (Level 4)
    l4 = comparison_df[comparison_df['Level'] == 'level4']
    if not l4.empty and 'R2_primary' in l4.columns:
        best_dyn = l4.loc[l4['R2_primary'].idxmax()]
        best['dynamic'] = {'model': best_dyn['Model'],
                           'R2': best_dyn['R2_primary'],
                           'level': 'Level 4'}

    return best


def generate_final_report(comparison_df: pd.DataFrame, best_models: Dict,
                           save_dir: Path):
    """Generate final benchmark report as CSV and summary figure."""
    # Save comparison table
    csv_path = save_dir / "full_benchmark_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"  [OK] Saved comparison table: {csv_path}")

    # Text summary
    report_lines = [
        "=" * 70,
        "  PHASE 3 — FINAL BENCHMARK REPORT",
        "=" * 70,
        "",
        "  BEST MODEL SELECTION",
        "  " + "-" * 40,
    ]
    for category, info in best_models.items():
        report_lines.append(f"  [BEST] Best {category.title()} Surrogate: "
                             f"{info['model']}  (R²={info['R2']:.4f})  [{info['level']}]")

    report_lines += [
        "",
        "  FULL COMPARISON TABLE",
        "  " + "-" * 40,
        comparison_df.to_string(index=False) if not comparison_df.empty else "  No results",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    report_path = save_dir / "final_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"  [OK] Saved report: {report_path}")

    # Best models JSON
    best_path = save_dir / "best_models.json"
    with open(best_path, 'w') as f:
        json.dump(best_models, f, indent=2, default=str)

    return report_text


def run_benchmarking() -> Dict:
    logger.info("=" * 70)
    logger.info("  BENCHMARKING — BEST MODEL SELECTION")
    logger.info("=" * 70)

    all_results = load_all_benchmarks()
    comparison_df = extract_key_metrics(all_results)
    best_models = select_best_models(comparison_df)
    report = generate_final_report(comparison_df, best_models, BENCH_DIR)

    print("\n" + report)

    return {'comparison': comparison_df, 'best': best_models}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    run_benchmarking()
