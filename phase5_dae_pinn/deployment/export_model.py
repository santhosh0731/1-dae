"""
ONNX and TorchScript Exporter
==============================
Prepares the DAE-PINN model for deployment in Digital Twin environments.
"""

import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_onnx_and_script(
    model,
    export_dir: Path,
) -> None:
    """Export model to ONNX and TorchScript formats."""
    export_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Create dummy input representing one batch sample: [t, Vin, D, Fs, L, C, Rload]
    dummy_input = torch.randn(1, 7, device=next(model.parameters()).device)

    # 1. Export to TorchScript
    try:
        script_module = torch.jit.trace(model, dummy_input)
        script_path = export_dir / "dae_pinn_model.pt"
        script_module.save(script_path)
        logger.info(f"  [EXPORT] TorchScript saved: {script_path.name}")
    except Exception as e:
        logger.warning(f"  TorchScript export failed: {e}")

    # 2. Export to ONNX
    try:
        onnx_path = export_dir / "dae_pinn_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['inputs'],
            output_names=['predictions'],
            dynamic_axes={'inputs': {0: 'batch_size'}, 'predictions': {0: 'batch_size'}}
        )
        logger.info(f"  [EXPORT] ONNX saved: {onnx_path.name}")
    except Exception as e:
        logger.warning(f"  ONNX export failed: {e}")
