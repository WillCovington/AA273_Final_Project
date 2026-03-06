import numpy as np

# importing everything over
from ground_stations import define_ground_station_locations
from ekf_fun import build_R_full, take_measurements
from ukf_fun import ukf_run
from gravity.dynamics import make_time_grid, rollout
from gravity.gravity_model import GravityModel
from plot import plot_truth_vs_est, plot_trajectory_with_moon, plot_ground_track
from save_run import *


def main():
    date_str = "03-06-2026"   # update as needed
    seed = 134
    rng = np.random.default_rng(seed)

    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")

    ##################### SETTING UP OUR TRUTH MODEL #####################
    alt_km = 300.0
    r_mag = model.r0_m + alt_km * 1000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)
    T_period = 2 * np.pi * np.sqrt(r_mag**3 / mu)   # orbital period [s]
    prop_duration = 0.5   # number of orbital periods

    # truth initial state
    x0_truth = np.array([r_mag, 0.0, 0.0, 0.0, v_circ / 2.0, v_circ], dtype=np.float64)

    # time grid 
    dt = 5.0 # step propogation [s]
    t_grid = make_time_grid(0.0, T_period * prop_duration, 5.0)

    L_truth = model.lmax_data

    # truth trajectory
    X_truth = rollout(x0_truth, t_grid, L_truth, model)
    print("---------------------------Truth Propagation Finished---------------------------")

    ##################### SETTING UP OUR MEASUREMENTS #####################

    gs_locations = define_ground_station_locations(n=20, lat_max_deg=45, seed=seed)
    sigma_rho = 5.0       # m
    sigma_rhodot = 0.05   # m/s
    elev_mask = 5.0       # deg

    R_full = build_R_full(len(gs_locations), sigma_rho, sigma_rhodot)

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

    print("---------------------------Truth Measurements Finished---------------------------")

    ##################### RUNNING THROUGH OUR UKF ESTIMATES AT VARYING L #####################

    # UKF initial guess
    x0 = x0_truth.copy()
    P0 = np.diag([1e6, 1e6, 1e6, 1e0, 1e0, 1e0])

    # process noise
    Q = np.eye(6) * 1e-6

    # UKF sigma-point parameters
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0

    L_list = [2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600]

    for L_max in L_list:
        runname = f"ukf_sweep_L{L_max}_Ngs{len(gs_locations)}_alt{alt_km}_T{prop_duration}per_seed{seed}"

        print("\n==============================")
        print("Running UKF with L_max =", L_max)

        ts, Xhat, Phat, timing = ukf_run(
            x0=x0,
            P0=P0,
            measurements=measurements,
            model=model,
            L_max=L_max,
            Q=Q,
            R_full=R_full,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

        meta = {
            "run_name": runname,
            "filter_type": "UKF",
            "seed": int(seed),
            "alt_km": float(alt_km),
            "prop_duration_periods": float(prop_duration),
            "dt_s": 1.0,
            "Ngs": int(len(gs_locations)),
            "sigma_rho": float(sigma_rho),
            "sigma_rhodot": float(sigma_rhodot),
            "elev_mask_deg": float(elev_mask),
            "L_truth": int(L_truth),
            "L_max": int(L_max),
            "Q_diag": np.diag(Q).tolist(),
            "alpha": float(alpha),
            "beta": float(beta),
            "kappa": float(kappa),
        }

        npz_path, fig_dir = save_run(
            date_str,
            runname,
            ts,
            X_truth,
            Xhat,
            Phat,
            meta,
            timing=timing,   # no timing unless you implement ukf_run_timed
        )

        print("Saved:", npz_path)
        print("Timing mean total step [ms]:", 1e3 * float(np.mean(timing["total_step_s"])))

        # save figures
        plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error=True, save_dir=fig_dir)
        plot_trajectory_with_moon(X_truth, Xhat, model, save_dir=fig_dir)
        plot_ground_track(ts, X_truth, Xhat, gs_locations = gs_locations, truth_cmap="viridis", est_cmap="plasma", show_colorbar=True, save_dir=fig_dir)


if __name__ == "__main__":
    main()