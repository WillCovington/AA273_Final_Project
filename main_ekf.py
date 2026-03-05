import numpy as np

# importing everything over
from ground_stations import define_ground_station_locations
from ekf_fun import *
from gravity.dynamics import *
from gravity.gravity_model import *
from plot import *
from save_run import save_ekf_run

# if this works how I think it should, we only need to propogate the dynamics once before we get into doing the filter for each L truncation
def main():
    date_str = "03-05-2026" # NOTE: we gotta update this every day
    seed = 134
    rng = np.random.default_rng(seed)

    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")


    ##################### SETTING UP OUR TRUTH MODEL #####################
    alt_km = 50.0
    r_mag = model.r0_m + alt_km * 1000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)
    T_period = 2 * np.pi * np.sqrt(r_mag**3 / mu) # time for one orbital period, in seconds
    prop_duration = 5.0 # how many periods we want to propogate over SET TO 5 LATER

    # truth initial state
    x0_truth = np.array([r_mag, 0.0, 0.0, 0.0, 0.0, v_circ], dtype=np.float64)
    
    # time grid (1 Hz)
    t_grid = make_time_grid(0.0, T_period * prop_duration, 1.0) # depending on how long we want to simulate over, multiply T_period accordingly

    L_truth = model.lmax_data

    # truth trajectory
    X_truth = rollout(x0_truth, t_grid, L_truth, model)
    print("---------------------------Truth Propogation Finished---------------------------")

    ##################### SETTING UP OUR MEASUREMENTS #####################
    
    # stations + measurement noise
    gs_locations = define_ground_station_locations(n=10, lat_max_deg=45, seed=seed)
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

    print("---------------------------Truth Measurements Finished---------------------------")

    ##################### RUNNING THROUGH OUR EKF ESTIMATES AT VARYING L #####################

    # EKF initial guess (optionally perturbed)
    x0 = x0_truth.copy()
    P0 = np.diag([1e6, 1e6, 1e6, 1e0, 1e0, 1e0])

    # process noise (placeholder; tune later)
    Q = np.eye(6) * 1e-6

    L_list = [2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600]

    # iterating through our EKF's
    for L_max in L_list:
        runname = f"sweep_L{L_max}_Ngs{len(gs_locations)}_alt{alt_km}_T{prop_duration}per_seed{seed}"
        print("\n==============================")
        print("Running EKF with L_max =", L_max)
    
        ts, Xhat, Phat, timing = ekf_run(
                x0, P0, measurements, model,
                L_max=L_max,
                Q=Q, R_full=R_full,
                eps_fd=5.0
            )
        
        meta = {
            "run_name": runname,
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
        }

        npz_path, fig_dir = save_ekf_run(
                date_str, runname,
                ts, X_truth, Xhat, Phat,
                meta,
                timing=timing
            )

        print("Saved:", npz_path)
        print("Timing mean total step [ms]:", 1e3 * float(np.mean(timing["total_step_s"])))
    
        # now we plot everything and save the images as png's
        plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error = True, save_dir=fig_dir)
        plot_trajectory_with_moon(X_truth, Xhat, model, save_dir=fig_dir)
        plot_ground_track(ts, X_truth, Xhat, save_dir=fig_dir)
    
if __name__ == "__main__":
    main()