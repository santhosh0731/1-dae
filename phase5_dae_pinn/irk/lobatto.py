"""
Lobatto IIIA Implicit Runge-Kutta Solver (3-Stage, Order 4)
=============================================================
"""

import numpy as np
from typing import Callable, Tuple


def lobatto_step(
    f: Callable, t: float, y: np.ndarray, h: float,
    n_newton: int = 10, **kwargs
) -> np.ndarray:
    c = [0.0, 0.5, 1.0]
    a = [
        [0.0, 0.0, 0.0],
        [5.0/24.0, 1.0/3.0, -1.0/24.0],
        [1.0/6.0, 2.0/3.0, 1.0/6.0]
    ]
    b = [1.0/6.0, 2.0/3.0, 1.0/6.0]

    # Initialize implicit stages
    k = [f(t + c[i]*h, y, **kwargs) for i in range(3)]

    for _ in range(n_newton):
        k_new = []
        for i in range(3):
            # Compute stage values
            Y_i = y.copy()
            for j in range(3):
                Y_i = Y_i + h * a[i][j] * k[j]
            k_new.append(f(t + c[i]*h, Y_i, **kwargs))

        diff = sum(np.max(np.abs(k_new[i] - k[i])) for i in range(3))
        k = k_new
        if diff < 1e-10:
            break

    # Stiffly accurate step
    Y_next = y.copy()
    for i in range(3):
        Y_next = Y_next + h * b[i] * k[i]
    return Y_next


def solve_lobatto(
    f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
    n_steps: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    t_arr = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0
    for i in range(n_steps):
        Y[i+1] = lobatto_step(f, t_arr[i], Y[i], h, **kwargs)
    return t_arr, Y
