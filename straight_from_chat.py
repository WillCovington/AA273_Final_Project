import numpy as np

from gravity.gravity_model import GravityModel
from gravity.dynamics import rollout, make_time_grid, propagate

# ----------------------------
# Ground stations
# ----------------------------

MOON_OMEGA_RAD_S = 2.6617e-6
T0_S = 0.0

def _rotz(theta: float) -> np.ndarray:
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def R_I_to_BF(t_s: float) -> np.ndarray:
    theta = MOON_OMEGA_RAD_S * (t_s - T0_S)
    return _rotz(theta)

def define_ground_station_locations(n, lat_max_deg = 30.0, seed=None):
    # equally longitudinally spaced ground with some variance in latitude
    rng = np.random.default_rng(seed)
    locations = []
    for j in range(n):
        lon = j * (360.0/n)
        lat = rng.uniform(-lat_max_deg, lat_max_deg)
        locations.append((lat, lon))
    return locations

def gs_bodyfixed_position(gs_loc, r_moon):
    lat_deg, lon_deg = gs_loc
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    clat = np.cos(lat); slat = np.sin(lat)
    clon = np.cos(lon); slon = np.sin(lon)

    return r_moon * np.array([clat*clon, clat*slon, slat], dtype=np.float64)

def gs_state_inertial(gs_loc, t_s, model):
    r_gs_BF = gs_bodyfixed_position(gs_loc, model.r0_m)

    R_I2BF = R_I_to_BF(t_s)
    R_BF2I = R_I2BF.T

    r_gs_I = R_BF2I @ r_gs_BF

    omega_I = np.array([0.0, 0.0, MOON_OMEGA_RAD_S], dtype=np.float64)
    v_gs_I = np.cross(omega_I, r_gs_I)
    return r_gs_I, v_gs_I

def is_visible_from_station(r_sc_I, r_gs_I, elev_mask_deg=0.0):
    los = r_sc_I - r_gs_I
    u_los = los / (np.linalg.norm(los) + 1e-12)
    u_up = r_gs_I / (np.linalg.norm(r_gs_I) + 1e-12)

    # elevation = asin(u_los · u_up)
    sin_el = float(np.dot(u_los, u_up))
    sin_el = np.clip(sin_el, -1.0, 1.0)
    el_deg = np.degrees(np.arcsin(sin_el))
    return el_deg >= elev_mask_deg

# ----------------------------
# Measurement model
# ----------------------------

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

# ----------------------------
# Dynamics Jacobian for covariance propagation (finite diff)
# ----------------------------

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

# ----------------------------
# EKF
# ----------------------------

def ekf_run(x0, P0, measurements, model, L_max, Q, R_full, eps_fd=5.0):
    x = np.asarray(x0, dtype=np.float64).reshape(6,)
    P = np.asarray(P0, dtype=np.float64).reshape(6, 6)

    ts = [measurements[0]["t"]]
    xs = [x.copy()]
    Ps = [P.copy()]

    t_prev = float(measurements[0]["t"])
    gs_all = measurements[0]["gs"]

    for meas in measurements[1:]:
        t_k = float(meas["t"])
        dt = t_k - t_prev
        y_k = np.asarray(meas["y"], dtype=np.float64).reshape(-1,)

        # Predict state
        x_pred = propagate(x, t_prev, dt, L_max, model)

        # Predict covariance
        F = F_discrete(dt, t_k, x_pred, L_max, model, eps=eps_fd)
        P_pred = F @ P @ F.T + Q

        # Predict measurement + Jacobian
        # (noise-free predicted measurement)
        y_hat = take_measurements(x_pred, gs_all, t_k, model, sigma_rho=0.0, sigma_rhodot=0.0, add_noise=False)
        C = calculate_C_gs(gs_all, x_pred, t_k, model)

        filt = filter_valid(y_k, y_hat, C, R_full)
        if filt is None:
            # no measurement update
            x, P = x_pred, P_pred
        else:
            yv, yhatv, Cv, Rv = filt
            innov = yv - yhatv
            S = Cv @ P_pred @ Cv.T + Rv
            K = P_pred @ Cv.T @ np.linalg.inv(S)
            x = x_pred + K @ innov
            P = (np.eye(6) - K @ Cv) @ P_pred

        ts.append(t_k)
        xs.append(x.copy())
        Ps.append(P.copy())
        t_prev = t_k

    return np.array(ts), np.vstack(xs), Ps

# ----------------------------
# MAIN
# ----------------------------

def main():
    seed = 134
    rng = np.random.default_rng(seed)

    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")

    alt_km = 50.0
    r_mag = model.r0_m + alt_km * 1000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)

    # truth initial state
    x0_truth = np.array([r_mag, 0.0, 0.0, 0.0, v_circ, 0.0], dtype=np.float64)

    # EKF initial guess (optionally perturbed)
    x0 = x0_truth.copy()
    P0 = np.diag([1e6, 1e6, 1e6, 1e0, 1e0, 1e0])

    # degrees
    L_truth = model.lmax_data
    L_max = 600

    # time grid (1 Hz)
    t_grid = make_time_grid(0.0, 600.0, 1.0)

    # truth trajectory
    X_truth = rollout(x0_truth, t_grid, L_truth, model)

    # stations + measurement noise
    gs_locations = define_ground_station_locations(n=20, lat_max_deg=45, seed=seed)
    print(gs_locations)
    sigma_rho = 5.0       # m
    sigma_rhodot = 0.05   # m/s
    elev_mask = 5.0       # deg

    R_full = build_R_full(len(gs_locations), sigma_rho, sigma_rhodot)

    # measurements list
    measurements = []
    for k, t in enumerate(t_grid):
        y = take_measurements(
            X_truth[k],
            gs_locations,
            float(t),
            model,
            sigma_rho=sigma_rho,
            sigma_rhodot=sigma_rhodot,
            elev_mask_deg=elev_mask,
            add_noise=True,
            rng=rng
        )
        measurements.append({"t": float(t), "y": y, "gs": gs_locations})

    # process noise (placeholder; tune later)
    Q = np.eye(6) * 1e-6

    # run EKF
    ts, Xhat, Phat = ekf_run(x0, P0, measurements, model, L_max, Q, R_full, eps_fd=5.0)

    print("Final estimate:", Xhat[-1])
    print("Final truth   :", X_truth[-1])
    print("Final pos err [m]:", np.linalg.norm(Xhat[-1,:3] - X_truth[-1,:3]))
    print("Final vel err [m/s]:", np.linalg.norm(Xhat[-1,3:] - X_truth[-1,3:]))

if __name__ == "__main__":
    main()