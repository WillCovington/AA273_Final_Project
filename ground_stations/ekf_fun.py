import numpy as np
from ground_stations.ground_stations import *
from gravity.dynamics import *
from gravity.gravity_model import *
import time

# the majority of our ekf functions

def ekf_run(x0, P0, measurements, model, L_max, Q, R_full, eps_fd=5.0):
    x = np.asarray(x0, dtype=np.float64).reshape(6,)
    P = np.asarray(P0, dtype=np.float64).reshape(6, 6)

    ts = [measurements[0]["t"]]
    xs = [x.copy()]
    Ps = [P.copy()]

    t_prev = float(measurements[0]["t"])
    gs_all = measurements[0]["gs"]
    
    # as another means of trying to quantify how well the truncated EKF works, we should time each step
    timing = {
        "propagate_s": [],
        "F_s": [],
        "meas_model_s": [],
        "update_s": [],
        "total_step_s": [],
        "n_valid_meas": [],
    }

    for meas in measurements[1:]:
        t_k = float(meas["t"])
        dt = t_k - t_prev
        y_k = np.asarray(meas["y"], dtype=np.float64).reshape(-1,)
        
        t0 = time.perf_counter()

        # Predict state
        tp0 = time.perf_counter()
        x_pred = propagate(x, t_prev, dt, L_max, model)
        tp1 = time.perf_counter()

        # Predict covariance
        tf0 = time.perf_counter()
        F = F_discrete(dt, t_k, x_pred, L_max, model, eps=eps_fd)
        P_pred = F @ P @ F.T + Q
        tf1 = time.perf_counter()

        # Predict measurement + Jacobian
        # (noise-free predicted measurement)
        tm0 = time.perf_counter()
        y_hat = take_measurements(x_pred, gs_all, t_k, model, sigma_rho=0.0, sigma_rhodot=0.0, add_noise=False)
        C = calculate_C_gs(gs_all, x_pred, t_k, model)
        tm1 = time.perf_counter()

        # and now our update step
        tu0 = time.perf_counter()
        filt = filter_valid(y_k, y_hat, C, R_full)
        if filt is None:
            # no measurement update
            x, P = x_pred, P_pred
            n_valid = 0
        else:
            yv, yhatv, Cv, Rv = filt
            n_valid = int(yv.size)
            innov = yv - yhatv
            S = Cv @ P_pred @ Cv.T + Rv
            K = P_pred @ Cv.T @ np.linalg.inv(S)
            x = x_pred + K @ innov
            P = (np.eye(6) - K @ Cv) @ P_pred
            
        tu1 = time.perf_counter()
        
        t1 = time.perf_counter()
        
        # adding all of our timing terms
        timing["propagate_s"].append(tp1 - tp0)
        timing["F_s"].append(tf1 - tf0)
        timing["meas_model_s"].append(tm1 - tm0)
        timing["update_s"].append(tu1 - tu0)
        timing["total_step_s"].append(t1 - t0)
        timing["n_valid_meas"].append(n_valid)
        
        ts.append(t_k)
        xs.append(x.copy())
        Ps.append(P.copy())
        t_prev = t_k
       
    # getting everything organized
    ts = np.asarray(ts, dtype=np.float64)
    Xhat = np.vstack(xs).astype(np.float64)
    Phat = np.stack(Ps, axis=0).astype(np.float64)
    
    for k in timing:
        timing[k] = np.asarray(timing[k], dtype=np.float64)

    return ts, Xhat, Phat, timing

def build_R_full(Ngs, sigma_rho, sigma_rhodot):
    R_full = np.zeros((2*Ngs, 2*Ngs), dtype=np.float64)
    for i in range(Ngs):
        R_full[2*i, 2*i] = sigma_rho**2
        R_full[2*i+1, 2*i+1] = sigma_rhodot**2
    return R_full

def take_measurements(state, gs_locations, time, model, sigma_rho, sigma_rhodot, elev_mask_deg=0.0, add_noise=True, rng=None):
    """
    Returns y: shape (2*Ngs,), [rho1,rhodot1,rho2,rhodot2,...] with NaNs for invisible.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)

    Ngs = len(gs_locations)
    y = np.full(2*Ngs, np.nan, dtype=np.float64)

    for i, gs_loc in enumerate(gs_locations):
        r_gs, v_gs = gs_state_inertial(gs_loc, time, model)

        if not is_visible_from_station(r, r_gs, elev_mask_deg=elev_mask_deg):
            continue

        rho_vec = r - r_gs
        rho = np.linalg.norm(rho_vec) + 1e-12
        u = rho_vec / rho
        vrel = v - v_gs
        rhodot = float(u @ vrel)

        if add_noise:
            rho += rng.normal(0.0, sigma_rho)
            rhodot += rng.normal(0.0, sigma_rhodot)

        y[2*i] = rho
        y[2*i+1] = rhodot

    return y

def calculate_C_gs(gs_locations, state_estimate, time, model):
    Ngs = len(gs_locations)
    C = np.zeros((2*Ngs, 6), dtype=np.float64)

    r = np.asarray(state_estimate[0:3], dtype=np.float64)
    v = np.asarray(state_estimate[3:6], dtype=np.float64)

    for i, gs_loc in enumerate(gs_locations):
        r_gs, v_gs = gs_state_inertial(gs_loc, time, model)

        rho_vec = r - r_gs
        rho = np.linalg.norm(rho_vec) + 1e-12
        u = rho_vec / rho
        vrel = v - v_gs

        # range row
        H_rho = np.zeros((1, 6))
        H_rho[0, 0:3] = u

        # range-rate row
        P = np.eye(3) - np.outer(u, u)
        dr = (P @ vrel) / rho

        H_rhodot = np.zeros((1, 6))
        H_rhodot[0, 0:3] = dr
        H_rhodot[0, 3:6] = u

        C[2*i,   :] = H_rho
        C[2*i+1, :] = H_rhodot

    return C

def filter_valid(y, y_hat, C, R_full):
    valid = np.isfinite(y)
    if not np.any(valid):
        return None
    return y[valid], y_hat[valid], C[valid, :], R_full[np.ix_(valid, valid)]

def accel_jacobian(r, t, L_max, model, eps=5.0):
    r = np.asarray(r, dtype=np.float64).reshape(3,)
    A = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        dr = np.zeros(3); dr[i] = eps
        ap = model.accel_inertial(r + dr, t, L_max)
        am = model.accel_inertial(r - dr, t, L_max)
        A[:, i] = (ap - am) / (2.0 * eps)
    return A

def F_discrete(dt, t, x_pred, L_max, model, eps=5.0):
    A_bl = accel_jacobian(x_pred[:3], t, L_max, model, eps=eps)
    I3 = np.eye(3)
    return np.block([[I3, dt*I3],
                     [dt*A_bl, I3]])