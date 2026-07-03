"""
Numerical IRK Solver Metrics
=============================
Measures the convergence of Implicit Runge-Kutta steps and solver residuals.
"""

import torch
import numpy as np
from typing import Dict

from phase5_dae_pinn.irk.embedded_irk import DifferentiableRadauIIALayer


def compute_irk_integration_metrics(
    model,
    test_loader,
    config: dict,
    device: torch.device,
    t_std: float,
) -> Dict[str, float]:
    """Evaluate embedded Radau-IIA integration residuals on the test set."""
    irk_layer = DifferentiableRadauIIALayer(
        model=model,
        step_size_h=config['irk'].get('step_size_h', 1e-5)
    ).to(device)

    model.eval()
    errors = []

    with torch.no_grad():
        for X, Y, P in test_loader:
            X = X.to(device)
            P = P.to(device)
            pred = model(X)
            irk_res = irk_layer(X, pred, P, t_std)
            errors.append(irk_res.cpu().numpy())

    all_err = np.vstack(errors)
    IL_err = all_err[:, 0]
    Vc_err = all_err[:, 1]

    return {
        'IRK_IL_residual_norm': float(np.sqrt(np.mean(IL_err**2))),
        'IRK_Vc_residual_norm': float(np.sqrt(np.mean(Vc_err**2))),
        'IRK_max_residual':      float(np.max(np.abs(all_err))),
    }
