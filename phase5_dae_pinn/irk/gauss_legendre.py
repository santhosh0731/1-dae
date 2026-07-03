"""
Gauss-Legendre Implicit Runge-Kutta Solver (2-Stage, Order 4)
==============================================================
"""

import numpy as np
from typing import Callable, Tuple


def gauss_legendre_step(
    f: Callable, t: float, y: np.ndarray, h: float,
    n_newton: int = 10, **kwargs
) -> np.ndarray:
    sqrt3 = np.sqrt(3.0)
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1 = b2 = 0.5

    # Simple fixed-point / Newton iteration to solve implicit stage equations
    k1 = f(t + c1*h, y, **kwargs)
    k2 = f(t + c2*h, y, **kwargs)

    for _ in range(n_newton):
        Y1 = y + h * (a11*k1 + a12*k2)
        Y2 = y + h * (a21*k1 + a22*k2)
        k1_new = f(t + c1*h, Y1, **kwargs)
        k2_new = f(t + c2*h, Y2, **kwargs)
        if np.max(np.abs(k1_new - k1)) + np.max(np.abs(k2_new - k2)) < 1e-10:
            break
        k1, k2 = k1_new, k2_new

    return y + h * (b1*k1 + b2*k2)


def solve_gauss_legendre(
    f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
    n_steps: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    t_arr = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0
    for i in range(n_steps):
        Y[i+1] = gauss_legendre_step(f, t_arr[i], Y[i], h, **kwargs)
    return t_arr, Y
