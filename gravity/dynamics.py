"""
gravity/dynamics.py

given r,t,L what gravity acceleration do I feel?
this file uses that acceleration repeatedly to propagate the full 6D state:

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