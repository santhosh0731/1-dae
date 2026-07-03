"""
Backward Differentiation Formula (BDF) Solver
==============================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple


def solve_bdf(f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
              t_eval: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        lambda t, y: f(t, y, **kwargs),
        t_span, y0,
        method='BDF',
        t_eval=t_eval,
        rtol=1e-6, atol=1e-9
    )
    return sol.t, sol.y.T
