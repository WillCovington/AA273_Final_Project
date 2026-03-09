import numpy as np
import time

from gravity.dynamics import propagate
from gnss.gnss_measurements import *


# same as our ground station jacobians
def accel_jacobian(r, t, L_max, model, eps=5.0):
    r = np.asarray(r, dtype=np.float64).reshape(3,)
    A = np.zeros((3, 3), dtype=np.float64)

    for i in range(3):
        dr = np.zeros(3, dtype=np.float64)
        dr[i] = eps

        ap = model.accel_inertial(r + dr, t, L_max)
        am = model.accel_inertial(r - dr, t, L_max)

        A[:, i] = (ap - am) / (2.0 * eps)

    return A


# calculates our A Jacobian (also called F) using the same discretization technique as the 
def F_discrete(dt, t, x_pred, L_max, model, eps=5.0):
    A_bl = accel_jacobian(x_pred[:3], t, L_max, model, eps=eps)
    I3 = np.eye(3)
    return np.block([
        [I3,       dt * I3],
        [dt*A_bl,  I3     ]
    ])

# damn near identical to our ground station ekf_run function
def ekf_run_gnss(
    x0,
    P0,
    measurements,
    model,
    L_max,
    Q,
    R_full,
    eps_fd=5.0,
):
    x = np.asarray(x0, dtype=np.float64).reshape(6,)
    P = np.asarray(P0, dtype=np.float64).reshape(6, 6)

    ts = [float(measurements[0]["t"])]
    xs = [x.copy()]
    Ps = [P.copy()]

    timing = {
        "propagate_s": [],
        "F_s": [],
        "meas_model_s": [],
        "update_s": [],
        "total_step_s": [],
        "n_valid_meas": [],
    }

    t_prev = float(measurements[0]["t"])
    gnss_all = measurements[0]["gnss"]

    for meas in measurements[1:]:
        t_k = float(meas["t"])
        dt = t_k - t_prev
        y_k = np.asarray(meas["y"], dtype=np.float64).reshape(-1,)

        t0 = time.perf_counter()

        # ---- Predict state ----
        tp0 = time.perf_counter()
        x_pred = propagate(x, t_prev, dt, L_max, model, method="rk4", substeps=1)
        tp1 = time.perf_counter()

        # ---- Predict covariance ----
        tf0 = time.perf_counter()
        F = F_discrete(dt, t_k, x_pred, L_max, model, eps=eps_fd)
        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        tf1 = time.perf_counter()

        # ---- Predict measurement + Jacobian ----
        tm0 = time.perf_counter()
        y_hat = take_gnss_measurements(
            x_pred,
            gnss_all,
            t_k,
            sigma_rho=0.0,
            sigma_rhodot=0.0,
            add_noise=False,
        )
        C = calculate_C_gnss(gnss_all, x_pred, t_k)
        tm1 = time.perf_counter()

        # ---- Update ----
        tu0 = time.perf_counter()
        filt = filter_valid_gnss(y_k, y_hat, C, R_full)

        if filt is None:
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
            P = 0.5 * (P + P.T)

        tu1 = time.perf_counter()
        t1 = time.perf_counter()

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

    ts = np.asarray(ts, dtype=np.float64)
    Xhat = np.vstack(xs).astype(np.float64)
    Phat = np.stack(Ps, axis=0).astype(np.float64)

    for k in timing:
        timing[k] = np.asarray(timing[k], dtype=np.float64)

    return ts, Xhat, Phat, timing