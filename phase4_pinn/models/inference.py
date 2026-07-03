"""
Phase 5 Export Pipeline
========================
Packages all Phase 4 outputs into a single directory
ready for Phase 5 DAE-PINN + Embedded IRK.

Exports:
  pinn_weights.pt           — best PINN model state dict
  physics_residuals.npz     — residuals on test set
  normalization_scalers.pkl — input/output scalers
  pinn_config.yaml          — hyperparameters
  phase4_metrics.json       — all benchmark results
  dae_formulation.py        — copy of DAE F(x,dx,z)=0
  solver_recommendation.txt — best solver for IRK
  README.txt                — Phase 5 loading instructions
"""

import os
import json
import shutil
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parents[2]
EXPORT_DIR  = BASE_DIR / "phase4_pinn" / "checkpoints" / "phase5_export"


def export_for_phase5(
    model,
    scalers:       Dict,
    metrics:       Dict,
    physics_residuals: Optional[np.ndarray] = None,
    solver_winner: str = "Radau-IIA",
) -> Path:
    """
    Create Phase 5 export package.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"export_{ts}"
    export_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n  Exporting Phase 5 package to: {export_path}")

    # 1. PINN weights
    weights_path = export_path / "pinn_weights.pt"
    torch.save({
        'model_state': model.state_dict(),
        'model_class': model.__class__.__name__,
        'input_dim':   model.input_dim,
        'output_dim':  model.output_dim,
        'output_names': model.OUTPUT_NAMES,
    }, weights_path)
    logger.info(f"    [OK] pinn_weights.pt")

    # 2. Scalers
    scalers_path = export_path / "normalization_scalers.pkl"
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"    [OK] normalization_scalers.pkl")

    # 3. Physics residuals
    if physics_residuals is not None:
        np.savez(export_path / "physics_residuals.npz", **physics_residuals)
        logger.info(f"    [OK] physics_residuals.npz")

    # 4. Metrics JSON
    with open(export_path / "phase4_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"    [OK] phase4_metrics.json")

    # 5. Config YAML
    config_src = BASE_DIR / "phase4_pinn" / "configs" / "pinn_config.yaml"
    if config_src.exists():
        shutil.copy(config_src, export_path / "pinn_config.yaml")
        logger.info(f"    [OK] pinn_config.yaml")

    # 6. DAE formulation
    dae_src = BASE_DIR / "phase4_pinn" / "physics" / "dae_formulation.py"
    if dae_src.exists():
        shutil.copy(dae_src, export_path / "dae_formulation.py")
        logger.info(f"    [OK] dae_formulation.py")

    # 7. Solver recommendation
    solver_txt = export_path / "solver_recommendation.txt"
    with open(solver_txt, 'w') as f:
        f.write(f"Recommended Numerical Solver for Phase 5 Embedded IRK\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Solver:    {solver_winner}\n")
        f.write(f"Reason:    Best accuracy for stiff boost converter DAE\n")
        f.write(f"Order:     3 (Radau-IIA 2-stage)\n")
        f.write(f"Stability: A-stable, L-stable\n")
        f.write(f"Reference: Hairer & Wanner, Solving ODEs II (1996)\n\n")
        f.write(f"Phase 5 will embed this as a differentiable IRK layer.\n")
    logger.info(f"    [OK] solver_recommendation.txt")

    # 8. README
    readme_path = export_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  PHASE 4 EXPORT — Ready for Phase 5 DAE-PINN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Export timestamp: {ts}\n\n")
        f.write("  FILES\n")
        f.write("  -----\n")
        f.write("  pinn_weights.pt           Load with torch.load()\n")
        f.write("  normalization_scalers.pkl Load with pickle.load()\n")
        f.write("  physics_residuals.npz     Load with np.load()\n")
        f.write("  phase4_metrics.json       Human-readable benchmark\n")
        f.write("  pinn_config.yaml          Hyperparameters\n")
        f.write("  dae_formulation.py        F(x,dx/dt,z)=0 equations\n")
        f.write("  solver_recommendation.txt Best solver for IRK\n\n")
        f.write("  LOADING IN PHASE 5\n")
        f.write("  ------------------\n")
        f.write("  import torch, pickle\n")
        f.write("  from phase4_pinn.models.pinn_model import PINNModel\n\n")
        f.write("  ckpt = torch.load('pinn_weights.pt')\n")
        f.write("  model = PINNModel(input_dim=7, output_dim=5)\n")
        f.write("  model.load_state_dict(ckpt['model_state'])\n\n")
        f.write("  with open('normalization_scalers.pkl','rb') as f:\n")
        f.write("      scalers = pickle.load(f)\n\n")
        f.write("  # Phase 5: extend model with IRK layer and DAE constraints\n")
        f.write("  # See dae_formulation.py for F(x,dx/dt,z)=0\n")

    logger.info(f"    [OK] README.txt")
    logger.info(f"\n  [DONE] Phase 5 export complete: {export_path}")
    return export_path
