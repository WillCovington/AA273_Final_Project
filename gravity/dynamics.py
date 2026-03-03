"""
gravity/dynamics.py

How motion evolves
Given (r, t, dt, L, model) → returns x_next
given r,t,L what gravity acceleration do I feel?

uses the acceleration to propagate the full 6D state:

    x = [rx, ry, rz, vx, vy, vz]

Main public calls:
- propagate(x0, t0, dt, L, model, ...)
- rollout(x0, t_grid, L, model, ...)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

# ============================================================
# SETTINGS
# ============================================================

STATE_DIM = 6

# Default integrator settings
DEFAULT_METHOD = "rk4"
DEFAULT_SUBSTEPS = 1  # can set >1 if want a smaller internal step for stability
EPS = 1e-15


# ============================================================
# State helpers
# ============================================================

def pack_state(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """pack r(3,) and v(3,) into x(6,)."""
    r = np.asarray(r, dtype=np.float64).reshape(3,)
    v = np.asarray(v, dtype=np.float64).reshape(3,)
    return np.hstack((r, v)).astype(np.float64)

def unpack_state(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """unpack x(6,) into r(3,) and v(3,)."""
    x = np.asarray(x, dtype=np.float64).reshape(STATE_DIM,)
    r = x[0:3]
    v = x[3:6]
    return r, v

def assert_state_shape(x: np.ndarray) -> None:
    """fail fast if x is not a valid 6D state."""
    x = np.asarray(x)
    if x.shape != (STATE_DIM,):
        raise ValueError(f"State must have shape (6,), got {x.shape}")

def assert_finite(x: np.ndarray, name: str = "x") -> None:
    """catch NaNs/Infs early so they don’t silently poison the simulation."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf.")
    
# ============================================================
# Continuous dynamics (ODE)
# ============================================================

def dynamics_rhs(t: float, x: np.ndarray, L: int, model) -> np.ndarray:
    """
    Continuous-time dynamics:
      r_dot = v
      v_dot = a(r,t;L)

    pull from the gravity model:
      a = model.accel_inertial(r, t, L)
    """
    assert_state_shape(x)
    r, v = unpack_state(x)

    a = model.accel_inertial(r, t, L)  # match gravity_model.py
    a = np.asarray(a, dtype=np.float64).reshape(3,)

    xdot = pack_state(v, a)
    return xdot

# ============================================================
# Integrator
# ============================================================

def step_rk4(t: float, x: np.ndarray, dt: float, L: int, model) -> np.ndarray:
    """
    RK4 (stable/simple for deterministic orbital propagation)
    """
    assert_state_shape(x)
    if dt <= 0:
        raise ValueError("dt must be > 0")

    k1 = dynamics_rhs(t, x, L, model)
    k2 = dynamics_rhs(t + 0.5 * dt, x + 0.5 * dt * k1, L, model)
    k3 = dynamics_rhs(t + 0.5 * dt, x + 0.5 * dt * k2, L, model)
    k4 = dynamics_rhs(t + dt, x + dt * k3, L, model)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(x_next, dtype=np.float64).reshape(STATE_DIM,)

# ============================================================
# Propagation API
# ============================================================

def propagate(
    x0: np.ndarray,
    t0: float,
    dt: float,
    L: int,
    model,
    method: str = DEFAULT_METHOD,
    substeps: int = DEFAULT_SUBSTEPS,
) -> np.ndarray:
    """
    propagate one time step

    Inputs:
      x0      : (6,) state [m, m/s]
      t0      : time [s]
      dt      : step size [s]
      L       : truncation degree
      model   : GravityModel instance (accel_inertial)
      method  : "rk4"
      substeps: if >1, split dt into smaller chunks for stability

    Output:
      x1 : (6,) next state
    """
    x = np.asarray(x0, dtype=np.float64).reshape(STATE_DIM,)
    assert_finite(x, "x0")

    if substeps < 1:
        raise ValueError("substeps must be >= 1")

    h = dt / float(substeps)
    t = float(t0)

    if method.lower() != "rk4":
        raise ValueError(f"Unknown method '{method}'. Only 'rk4' is implemented.")

    for _ in range(substeps):
        x = step_rk4(t, x, h, L, model)
        t += h

        # sanity check through every substep
        assert_finite(x, "x")

    return x


def rollout(
    x0: np.ndarray,
    t_grid: np.ndarray,
    L: int,
    model,
    method: str = DEFAULT_METHOD,
    substeps: int = DEFAULT_SUBSTEPS,
) -> np.ndarray:
    """
    I simulate a full trajectory over a provided time grid.

    Inputs:
      x0      : (6,) initial state
      t_grid  : (K+1,) times [s] (must be increasing)
      L       : truncation degree
      model   : GravityModel instance
      method/substeps : passed to propagate

    Output:
      X : (K+1, 6) stacked states, with X[0] = x0
    """
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1,)
    if t_grid.size < 2:
        raise ValueError("t_grid must have at least 2 time points.")
    if not np.all(np.diff(t_grid) > 0):
        raise ValueError("t_grid must be strictly increasing.")

    K = t_grid.size - 1
    X = np.zeros((K + 1, STATE_DIM), dtype=np.float64)

    x = np.asarray(x0, dtype=np.float64).reshape(STATE_DIM,)
    assert_finite(x, "x0")
    X[0, :] = x

    for k in range(K):
        t0 = float(t_grid[k])
        dt = float(t_grid[k + 1] - t_grid[k])

        x = propagate(x, t0, dt, L, model, method=method, substeps=substeps)
        X[k + 1, :] = x

    return X

# ============================================================
# utilities functions for checks
# ============================================================

def state_norms(x: np.ndarray) -> Tuple[float, float]:
    """I return (||r||, ||v||) for checks"""
    r, v = unpack_state(x)
    return float(np.linalg.norm(r)), float(np.linalg.norm(v))


def make_time_grid(t0: float, tf: float, dt: float) -> np.ndarray:
    """build an inclusive time grid [t0, t0+dt, ..., tf]."""
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if tf <= t0:
        raise ValueError("tf must be > t0")
    n = int(np.floor((tf - t0) / dt))
    t = t0 + dt * np.arange(n + 1, dtype=np.float64)
    if t[-1] < tf - 0.5 * dt:
        # include the last point if the rounding cut it off too early
        t = np.hstack([t, np.array([tf], dtype=np.float64)])
    else:
        t[-1] = tf
    return t

# ============================================================
# smoke test
# ============================================================

if __name__ == "__main__":
    # load gravity model
    # build a circular-ish initial condition at 50 km altitude
    # short propagation and print norms

    from gravity_model import GravityModel 

    model = GravityModel.from_npz()

    # Simple ICs:
    # position along +x, velocity along +y (standard "circular orbit" starter)
    r_mag = model.r0_m + 50_000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)

    r0 = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, v_circ, 0.0], dtype=np.float64)
    x0 = pack_state(r0, v0)

    # Simulate for ~10 minutes
    t0 = 0.0
    tf = 600.0
    dt = 5.0
    t_grid = make_time_grid(t0, tf, dt)

    # Use truth L as whatever the file supports (660)
    L = model.lmax_data

    X = rollout(x0, t_grid, L, model, method="rk4", substeps=1)

    rN, vN = state_norms(X[-1])
    print("Final ||r|| [m]:", rN)
    print("Final ||v|| [m/s]:", vN)
    print("Done.")