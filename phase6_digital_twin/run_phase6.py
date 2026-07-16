"""
run_phase6.py — Master Orchestration Script for Phase 6 Digital Twin
=====================================================================

Demonstrates the final Phase 6 deliverables: live digital twin dashboard
simulation and the SciML LLM engineering assistant.
"""

import sys
import os

# Force UTF-8 encoding for stdout on Windows to support emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    os.environ["PYTHONUTF8"] = "1"

import logging
import yaml
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from phase6_digital_twin.dashboard_simulator import DigitalTwinDashboard
from phase6_digital_twin.llm_assistant import LLMEngineeringAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 Digital Twin Simulation Dashboard")
    parser.add_argument('--vin', type=float, default=48.0, help='Input voltage (V) [default: 48.0]')
    parser.add_argument('--d', type=float, default=0.6, help='Duty cycle (0.1-0.95) [default: 0.6]')
    parser.add_argument('--fs', type=float, default=50000.0, help='Switching frequency (Hz) [default: 50000.0]')
    parser.add_argument('--L', type=float, default=50e-6, help='Inductance (H) [default: 50 uH]')
    parser.add_argument('--C', type=float, default=47e-6, help='Capacitance (F) [default: 47 uF]')
    parser.add_argument('--rload', type=float, default=5.0, help='Load resistance (Ohm) [default: 5.0]')
    parser.add_argument('-i', '--interactive', action='store_true', help='Prompt interactively for parameters')
    args = parser.parse_args()

    # Check for interactive prompt
    if args.interactive:
        print("\n=== DIGITAL TWIN INTERACTIVE CONFIGURATION ===")
        try:
            val = input("Enter Input Voltage (Vin, Volts) [default: 48.0]: ").strip()
            vin = float(val) if val else 48.0

            val = input("Enter Duty Cycle (D, 0.1 - 0.95) [default: 0.6]: ").strip()
            d = float(val) if val else 0.6

            val = input("Enter Switching Frequency (Fs, Hz) [default: 50000]: ").strip()
            fs = float(val) if val else 50000.0

            val = input("Enter Inductance (L, Henries) [default: 50e-6]: ").strip()
            L = float(val) if val else 50e-6

            val = input("Enter Capacitance (C, Farads) [default: 47e-6]: ").strip()
            C = float(val) if val else 47e-6

            val = input("Enter Load Resistance (Rload, Ohms) [default: 5.0]: ").strip()
            rload = float(val) if val else 5.0
        except ValueError:
            print("[WARN] Invalid entry detected. Falling back to default values.")
            vin, d, fs, L, C, rload = 48.0, 0.6, 50000.0, 50e-6, 47e-6, 5.0
    else:
        vin, d, fs, L, C, rload = args.vin, args.d, args.fs, args.L, args.C, args.rload

    logger.info("+" + "="*63 + "+")
    logger.info("|  PHASE 6 — DIGITAL TWIN & LLM ENGINEERING ASSISTANT          |")
    logger.info("|  Final Proposed Edge Deployment & User Dashboard             |")
    logger.info("+" + "="*63 + "+")

    # Load Phase 6 configuration
    cfg_path = BASE_DIR / "configs" / "phase6_config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Initialize dashboard simulator
    model_path = Path(config['paths']['torchscript_model'])
    scalers_path = Path(config['paths']['scalers'])

    # Fallback to general export paths if not found
    if not model_path.exists():
        model_path = BASE_DIR.parent / "phase5_dae_pinn" / "checkpoints" / "dae_pinn_best.pt"
        logger.warning(f"TorchScript model not found at {model_path}. Trying fallback...")
        if not model_path.exists():
            logger.error("No trained models found! Please run Phase 5 first.")
            return

    # Check scalers
    if not scalers_path.exists():
        export_dir = BASE_DIR.parent / "phase4_pinn" / "checkpoints" / "phase5_export"
        subdirs = list(export_dir.glob("export_*"))
        if subdirs:
            scalers_path = subdirs[-1] / "normalization_scalers.pkl"
        else:
            logger.error("No scalers found! Please run Phase 4 first.")
            return

    logger.info(f"Loading Digital Twin Model: {model_path.name}")
    logger.info(f"Loading Normalization Scalers: {scalers_path.name}")

    dashboard = DigitalTwinDashboard(model_path, scalers_path, BASE_DIR / "plots")

    # Run simulation for custom operating condition
    params = {
        'Vin':   vin,
        'D':     d,
        'Fs':    fs,
        'L':     L,
        'C':     C,
        'Rload': rload
    }
    logger.info(f"\n  Running Digital Twin transient simulation for: Vin={params['Vin']}V, D={params['D']}...")
    t, preds = dashboard.run_simulation(**params)

    # Plot dashboard telemetry
    plot_path = dashboard.plot_dashboard(t, preds, params)
    logger.info(f"  [OK] Dashboard telemetry saved: {plot_path}")

    # Load metrics for assistant
    metrics = {}
    metrics_file = BASE_DIR.parent / "phase5_dae_pinn" / "reports" / "phase5_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    # Conversational Assistant Mockup
    logger.info("\n" + "="*60 + "\n  SciML CONVERSATIONAL ENGINEERING ASSISTANT\n" + "="*60)
    assistant = LLMEngineeringAssistant(metrics)

    questions = [
        "What is the underlying DAE formulation of this model?",
        "Show me the model accuracy metrics and R2 values.",
        "Why did you recommend Radau-IIA as the numerical solver?",
    ]

    for q in questions:
        logger.info(f"\nUser Query: '{q}'")
        response = assistant.ask(q)
        print(response)

    logger.info("\n" + "="*60)
    logger.info("  [DONE] Phase 6 simulation dashboard completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
