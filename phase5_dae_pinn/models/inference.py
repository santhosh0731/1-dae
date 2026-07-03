"""
Inference & Prediction Loader
==============================
Exposes predictions and loads weights for DAE-PINN models.
"""

import pickle
import torch
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from phase5_dae_pinn.models.dae_pinn import DAEPINNModel


def load_dae_pinn_model(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[DAEPINNModel, Dict]:
    """Load model weight checkpoint and its configs."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    model = DAEPINNModel(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model'].get('dropout_rate', 0.05),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    return model, config
