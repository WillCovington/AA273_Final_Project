import numpy as np
from gravity.gravity_model import GravityModel
from gravity.dynamics import rollout, make_time_grid
from take_measurements import define_ground_station_locations, build_R_full, generate_measurements


# this is the ekf process used for estimating our satellite's state using ground stations on the moon
# once this is working, the plan is to move onto the UKF, then later onto using GNSS satellites
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
    gs_locations = define_ground_station_locations(n=30, lat_max_deg=30, seed=seed)
    print(gs_locations)
    sigma_rho = 5.0       # m
    sigma_rhodot = 0.05   # m/s
    elev_mask = 5.0       # deg

    R_full = build_R_full(len(gs_locations), sigma_rho, sigma_rhodot)

    # measurements list
    measurements = []
    for k, t in enumerate(t_grid):
        y = generate_measurements(
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