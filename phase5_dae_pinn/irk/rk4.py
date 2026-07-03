"""
Classic 4th-Order Runge-Kutta Solver (RK4)
===========================================
"""

import numpy as np
from typing import Callable, Tuple


def rk4_step(f: Callable, t: float, y: np.ndarray, h: float, **kwargs) -> np.ndarray:
    k1 = f(t,         y,              **kwargs)
    k2 = f(t + h/2,   y + h/2 * k1,  **kwargs)
    k3 = f(t + h/2,   y + h/2 * k2,  **kwargs)
    k4 = f(t + h,     y + h   * k3,  **kwargs)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def solve_rk4(f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
              n_steps: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    t_arr = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0
    for i in range(n_steps):
        Y[i+1] = rk4_step(f, t_arr[i], Y[i], h, **kwargs)
    return t_arr, Y
