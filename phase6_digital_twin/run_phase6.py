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
    logger.info("+" + "="*63 + "+")
    logger.info("|  PHASE 6 — DIGITAL TWIN & LLM ENGINEERING ASSISTANT          |")
    logger.info("|  Final Proposed Edge Deployment & User Dashboard             |")
    logger.info("+" + "="*63 + "+")

    # Load Phase 6 configuration
    cfg_path = BASE_DIR / "configs" / "phase6_config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Initialize dashboard simulator
    # Wait, lets check if the exported model and scalers exist from Phase 5
    model_path = Path(config['paths']['torchscript_model'])
    scalers_path = Path(config['paths']['scalers'])

    # Fallback to general export paths if not found
    if not model_path.exists():
        model_path = BASE_DIR.parent / "phase5_dae_pinn" / "checkpoints" / "dae_pinn_best.pt"
        # If it's the pt checkpoint we save model_state, so we can convert/use the best model checkpoint directly
        logger.warning(f"TorchScript model not found at {model_path}. Trying fallback...")
        if not model_path.exists():
            logger.error("No trained models found! Please run Phase 5 first.")
            return

    # Check scalers
    if not scalers_path.exists():
        # search under phase4 export
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

    # Run simulation for typical custom operating condition
    params = {
        'Vin':   48.0,
        'D':     0.6,
        'Fs':    50000.0,
        'L':     50e-6,
        'C':     47e-6,
        'Rload': 5.0
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
