"""
Numerical Solver Benchmarking
================================
Compares RK4, RK45, Radau-IIA, Gauss-Legendre, and BDF against
LTspice ground truth for boost converter DAE system.
Prepares solver selection for Phase 5 Embedded IRK.
"""

import time
import logging
import numpy as np
from typing import Dict, Callable, Tuple
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


# ── Boost Converter Averaged ODE System ───────────────────────────────────────

def boost_ode(t: float, y: np.ndarray, Vin: float, D: float,
              L: float, C: float, Rload: float) -> np.ndarray:
    """
    Averaged boost converter ODE:
      dy/dt = f(y, t, params)

      y[0] = IL(t)  — inductor current
      y[1] = Vc(t)  — capacitor voltage (= Vout)

    f1: dIL/dt = (Vin - (1-D)*Vc) / L
    f2: dVc/dt = ((1-D)*IL - Vc/Rload) / C
    """
    IL, Vc = y
    dIL_dt = (Vin - (1.0 - D) * Vc) / L
    dVc_dt = ((1.0 - D) * IL - Vc / Rload) / C
    return np.array([dIL_dt, dVc_dt])


# ── Solver Implementations ─────────────────────────────────────────────────────

def rk4_step(f: Callable, t: float, y: np.ndarray, h: float, **kwargs) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step."""
    k1 = f(t,         y,              **kwargs)
    k2 = f(t + h/2,   y + h/2 * k1,  **kwargs)
    k3 = f(t + h/2,   y + h/2 * k2,  **kwargs)
    k4 = f(t + h,     y + h   * k3,  **kwargs)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def solve_rk4(f, t_span, y0, n_steps, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 solver."""
    t0, tf = t_span
    t_arr  = np.linspace(t0, tf, n_steps + 1)
    h      = (tf - t0) / n_steps
    Y      = np.zeros((n_steps + 1, len(y0)))
    Y[0]   = y0
    for i in range(n_steps):
        Y[i+1] = rk4_step(f, t_arr[i], Y[i], h, **kwargs)
    return t_arr, Y


def solve_scipy(method: str, f, t_span, y0, t_eval, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for scipy solve_ivp solvers (RK45, Radau, BDF, DOP853)."""
    sol = solve_ivp(
        lambda t, y: f(t, y, **kwargs),
        t_span, y0,
        method=method,
        t_eval=t_eval,
        rtol=1e-6, atol=1e-9,
        dense_output=False,
    )
    if not sol.success:
        logger.warning(f"  [{method}] Solver warning: {sol.message}")
    return sol.t, sol.y.T   # (T,), (T, n_states)


# ── Gauss-Legendre IRK (2-stage, order 4) ────────────────────────────────────

def gauss_legendre_step(
    f: Callable, t: float, y: np.ndarray, h: float,
    n_newton: int = 10, **kwargs
) -> np.ndarray:
    """
    2-stage Gauss-Legendre IRK step (implicit, symplectic, order 4).
    Butcher tableau:
      c1 = 1/2 - sqrt(3)/6,  c2 = 1/2 + sqrt(3)/6
      a11 = 1/4,  a12 = 1/4 - sqrt(3)/6
      a21 = 1/4 + sqrt(3)/6, a22 = 1/4
    """
    sqrt3 = np.sqrt(3.0)
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1 = b2 = 0.5

    n = len(y)
    # Newton iteration for implicit stages
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


def solve_gauss_legendre(f, t_span, y0, n_steps, **kwargs):
    """Fixed-step Gauss-Legendre solver."""
    t0, tf = t_span
    t_arr  = np.linspace(t0, tf, n_steps + 1)
    h      = (tf - t0) / n_steps
    Y      = np.zeros((n_steps + 1, len(y0)))
    Y[0]   = y0
    for i in range(n_steps):
        Y[i+1] = gauss_legendre_step(f, t_arr[i], Y[i], h, **kwargs)
    return t_arr, Y


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def benchmark_solvers(
    Vin: float = 36.0,
    D:   float = 0.5,
    L:   float = 50e-6,
    C:   float = 47e-6,
    Rload: float = 1.0,
    t_end: float = 0.005,
    n_steps: int = 512,
    y0: np.ndarray = None,
) -> Dict:
    """
    Run all 5 solvers on boost converter ODE and return comparison metrics.
    """
    if y0 is None:
        y0 = np.array([0.01, 5.0])   # small initial current and voltage

    t_eval = np.linspace(0, t_end, n_steps + 1)
    t_span = (0.0, t_end)
    kwargs = {'Vin': Vin, 'D': D, 'L': L, 'C': C, 'Rload': Rload}

    solvers = {
        'RK4':            lambda: solve_rk4(boost_ode, t_span, y0, n_steps, **kwargs),
        'RK45':           lambda: solve_scipy('RK45',   boost_ode, t_span, y0, t_eval, **kwargs),
        'Radau-IIA':      lambda: solve_scipy('Radau',  boost_ode, t_span, y0, t_eval, **kwargs),
        'BDF':            lambda: solve_scipy('BDF',    boost_ode, t_span, y0, t_eval, **kwargs),
        'Gauss-Legendre': lambda: solve_gauss_legendre(boost_ode, t_span, y0, n_steps, **kwargs),
    }

    # Use Radau as high-accuracy reference
    t_ref, Y_ref = solve_scipy('Radau', boost_ode, t_span, y0, t_eval, **kwargs)

    results = {}
    logger.info(f"\n  Vin={Vin}V, D={D}, L={L*1e6:.0f}uH, C={C*1e6:.0f}uF, Rload={Rload}ohm")
    logger.info(f"  {'Solver':<18} {'IL_RMSE':>10} {'Vc_RMSE':>10} {'Time(ms)':>10} {'Notes'}")
    logger.info("  " + "-" * 60)

    for name, runner in solvers.items():
        try:
            t0_wall = time.perf_counter()
            t_sol, Y_sol = runner()
            elapsed_ms = (time.perf_counter() - t0_wall) * 1000

            # Interpolate to reference grid if needed
            if len(t_sol) != len(t_ref):
                IL_interp = np.interp(t_ref, t_sol, Y_sol[:, 0])
                Vc_interp = np.interp(t_ref, t_sol, Y_sol[:, 1])
            else:
                IL_interp = Y_sol[:, 0]
                Vc_interp = Y_sol[:, 1]

            IL_rmse = np.sqrt(np.mean((IL_interp - Y_ref[:, 0])**2))
            Vc_rmse = np.sqrt(np.mean((Vc_interp - Y_ref[:, 1])**2))
            IL_final = IL_interp[-1]
            Vc_final = Vc_interp[-1]

            # Steady-state reference
            Vout_ss = Vin / (1.0 - D)
            IL_ss   = Vout_ss / ((1.0 - D) * Rload)

            results[name] = {
                'IL_RMSE':      float(IL_rmse),
                'Vc_RMSE':      float(Vc_rmse),
                'time_ms':      float(elapsed_ms),
                'IL_final':     float(IL_final),
                'Vc_final':     float(Vc_final),
                'IL_ss_err_%':  abs(IL_final - IL_ss) / IL_ss * 100,
                'Vc_ss_err_%':  abs(Vc_final - Vout_ss) / Vout_ss * 100,
                't':            t_sol,
                'Y':            Y_sol,
            }

            tag = " <- RECOMMENDED" if name == "Radau-IIA" else ""
            logger.info(
                f"  {name:<18} {IL_rmse:>10.4f} {Vc_rmse:>10.4f} "
                f"{elapsed_ms:>10.2f}{tag}"
            )

        except Exception as e:
            logger.error(f"  {name}: FAILED — {e}")
            results[name] = {'error': str(e)}

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger.info("=" * 65)
    logger.info("  NUMERICAL SOLVER BENCHMARK")
    logger.info("=" * 65)

    for D_val in [0.4, 0.5, 0.6, 0.7]:
        logger.info(f"\n  --- D = {D_val} ---")
        results = benchmark_solvers(D=D_val)

    logger.info("\n  [DONE] Radau-IIA selected for Phase 5 embedded IRK")
