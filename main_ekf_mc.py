import numpy as np
from pathlib import Path

from ground_stations import define_ground_station_locations
from ground_stations.ekf_fun import *
from gravity.dynamics import *
from gravity.gravity_model import *
from analysis.plot import *
from analysis.save_run import *
from analysis.metrics import position_rmse, velocity_rmse #, nees_series

def nees_series(X_truth, Xhat, Phat):
    """
    Returns NEES[k] = e_k^T P_k^{-1} e_k
    """
    X_truth = np.asarray(X_truth)
    Xhat = np.asarray(Xhat)
    Phat = np.asarray(Phat)

    K = Xhat.shape[0]
    nees = np.zeros(K, dtype=np.float64)

    for k in range(K):
        e = (X_truth[k] - Xhat[k]).reshape(6, 1)
        P = Phat[k]
        nees[k] = (e.T @ np.linalg.solve(P, e)).item()

    return nees

def build_measurements(X_truth, t_grid, gs_locations, model, sigma_rho, sigma_rhodot,
                       elev_mask, rng):
    """
    Build one noisy measurement list for a full truth trajectory
    """
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
    return measurements


def sample_initial_estimate(x0_truth, P0, rng):
    """
    Sample an initial estimate from the prior covariance
    """
    dx0 = rng.multivariate_normal(mean=np.zeros(6), cov=P0)
    return x0_truth + dx0


def save_mc_summary(
    date_str,
    sweep_name,
    summary_dict,
    meta,
    out_root="runs"
):
    """
    Save MC sweep summary to one compressed npz
    """
    out_root = Path(out_root)
    sweep_dir = out_root / date_str / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    save_path = sweep_dir / f"{sweep_name}_mc_summary.npz"

    np.savez_compressed(
        save_path,
        **summary_dict,
        meta_json=np.array(str(meta), dtype=object)
    )

    return str(save_path)


def main():
    date_str = "03-15-2026"   # update as needed
    sweep_name = "ekf_mc_sweep_50km_1orbit"

    # ============================================================
    # MONTE CARLO SETTINGS
    # ============================================================

    seed = 134
    N_mc = 3   # 50 or 100 for valid results
    save_individual_runs = False   # set True to save every MC run separately
    make_individual_plots = False  # keep False for speed

    # Model

    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")

    # Truth Model

    alt_km = 50.0
    r_mag = model.r0_m + alt_km * 1000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)
    T_period = 2 * np.pi * np.sqrt(r_mag**3 / mu)
    prop_duration = 1.0   # set to 5.0 later for final runs!!!!!!!!

    # truth initial state
    x0_truth = np.array([0.0, r_mag, 0.0, 0.0, 0.0, v_circ], dtype=np.float64)

    # time grid
    dt = 30.0
    t_grid = make_time_grid(0.0, T_period * prop_duration, dt)

    L_truth = model.lmax_data

    # truth trajectory (same for all MC runs)
    X_truth = rollout(x0_truth, t_grid, L_truth, model)
    print("---------------------------Truth Propagation Finished---------------------------")

    # ============================================================
    # MEASUREMENT SETUP
    # ============================================================

    gs_locations = define_ground_station_locations(n=10, lat_max_deg=45, seed=seed)

    sigma_rho = 5.0       # m
    sigma_rhodot = 0.05   # m/s
    elev_mask = 5.0       # deg

    R_full = build_R_full(len(gs_locations), sigma_rho, sigma_rhodot)

    # ============================================================
    # FILTER SETTINGS
    # ============================================================

    # prior covariance
    P0 = np.diag([1e6, 1e6, 1e6, 1e0, 1e0, 1e0])

    # process noise
    #Q = np.eye(6) * 1e-6
    Q = np.diag([1e-10, 1e-10, 1e-10, 1e-6, 1e-6, 1e-6])

    # truncation sweep
    L_list = [5, 10, 50, 100, 200, 300, 400, 500, 600]
    # L_list = [2, 10, 50] # short list for quick testing

    # ============================================================
    # MC SUMMARY STORAGE
    # ============================================================

    nL = len(L_list)
    K = len(t_grid)

    pos_rmse_xyz_mean = np.zeros((nL, 3), dtype=np.float64)
    pos_rmse_xyz_std = np.zeros((nL, 3), dtype=np.float64)
    pos_rmse_norm_mean = np.zeros(nL, dtype=np.float64)
    pos_rmse_norm_std = np.zeros(nL, dtype=np.float64)

    vel_rmse_xyz_mean = np.zeros((nL, 3), dtype=np.float64)
    vel_rmse_xyz_std = np.zeros((nL, 3), dtype=np.float64)
    vel_rmse_norm_mean = np.zeros(nL, dtype=np.float64)
    vel_rmse_norm_std = np.zeros(nL, dtype=np.float64)

    nees_mean = np.zeros((nL, K), dtype=np.float64)
    nees_std = np.zeros((nL, K), dtype=np.float64)

    timing_mean_ms = np.zeros(nL, dtype=np.float64)
    timing_std_ms = np.zeros(nL, dtype=np.float64)

    # ============================================================
    # MONTE CARLO SWEEP
    # ============================================================

    for iL, L_max in enumerate(L_list):
        print("\n============================================================")
        print(f"Running EKF Monte Carlo sweep for L_max = {L_max}")
        print("============================================================")

        pos_rmse_xyz_runs = np.zeros((N_mc, 3), dtype=np.float64)
        pos_rmse_norm_runs = np.zeros(N_mc, dtype=np.float64)

        vel_rmse_xyz_runs = np.zeros((N_mc, 3), dtype=np.float64)
        vel_rmse_norm_runs = np.zeros(N_mc, dtype=np.float64)

        nees_runs = np.zeros((N_mc, K), dtype=np.float64)
        timing_runs_ms = np.zeros(N_mc, dtype=np.float64)

        for mc in range(N_mc):
            seed_mc = seed + 1000 * iL + mc
            rng = np.random.default_rng(seed_mc)

            print(f"  MC run {mc+1:02d}/{N_mc:02d} | seed = {seed_mc}")

            # noisy measurements for this run
            measurements = build_measurements(
                X_truth=X_truth,
                t_grid=t_grid,
                gs_locations=gs_locations,
                model=model,
                sigma_rho=sigma_rho,
                sigma_rhodot=sigma_rhodot,
                elev_mask=elev_mask,
                rng=rng
            )

            # perturbed initial estimate
            x0 = sample_initial_estimate(x0_truth, P0, rng)

            # EKF run
            ts, Xhat, Phat, timing = ekf_run(
                x0=x0,
                P0=P0,
                measurements=measurements,
                model=model,
                L_max=L_max,
                Q=Q,
                R_full=R_full,
                eps_fd=5.0
            )

            # metrics
            rmse_pos_xyz, rmse_pos_norm = position_rmse(X_truth, Xhat)
            rmse_vel_xyz, rmse_vel_norm = velocity_rmse(X_truth, Xhat)
            nees_k = nees_series(X_truth, Xhat, Phat)

            pos_rmse_xyz_runs[mc, :] = rmse_pos_xyz
            pos_rmse_norm_runs[mc] = rmse_pos_norm

            vel_rmse_xyz_runs[mc, :] = rmse_vel_xyz
            vel_rmse_norm_runs[mc] = rmse_vel_norm

            nees_runs[mc, :] = nees_k
            timing_runs_ms[mc] = 1e3 * float(np.mean(timing["total_step_s"]))

            # optionally save each individual MC run change to True above
            if save_individual_runs:
                runname = (
                    f"ekf_mc_L{L_max}_run{mc:03d}"
                    f"_Ngs{len(gs_locations)}_alt{alt_km}_T{prop_duration}per_seed{seed_mc}"
                )

                meta = {
                    "run_name": runname,
                    "filter_type": "EKF",
                    "is_monte_carlo": True,
                    "mc_index": int(mc),
                    "seed": int(seed_mc),
                    "N_mc": int(N_mc),
                    "alt_km": float(alt_km),
                    "prop_duration_periods": float(prop_duration),
                    "dt_s": float(dt),
                    "Ngs": int(len(gs_locations)),
                    "sigma_rho": float(sigma_rho),
                    "sigma_rhodot": float(sigma_rhodot),
                    "elev_mask_deg": float(elev_mask),
                    "L_truth": int(L_truth),
                    "L_max": int(L_max),
                    "Q_diag": np.diag(Q).tolist(),
                }

                npz_path, fig_dir = save_run(
                    date_str=date_str,
                    runname=runname,
                    ts=ts,
                    X_truth=X_truth,
                    Xhat=Xhat,
                    Phat=Phat,
                    meta=meta,
                    timing=timing
                )

                if make_individual_plots:
                    plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error=True, save_dir=fig_dir)
                    plot_trajectory_with_moon(X_truth, Xhat, model, save_dir=fig_dir)
                    plot_ground_track(
                        ts, X_truth, Xhat,
                        gs_locations=gs_locations,
                        truth_cmap="viridis",
                        est_cmap="plasma",
                        show_colorbar=True,
                        save_dir=fig_dir
                    )

        # aggregate over MC runs
        pos_rmse_xyz_mean[iL, :] = np.mean(pos_rmse_xyz_runs, axis=0)
        pos_rmse_xyz_std[iL, :] = np.std(pos_rmse_xyz_runs, axis=0)
        pos_rmse_norm_mean[iL] = np.mean(pos_rmse_norm_runs)
        pos_rmse_norm_std[iL] = np.std(pos_rmse_norm_runs)

        vel_rmse_xyz_mean[iL, :] = np.mean(vel_rmse_xyz_runs, axis=0)
        vel_rmse_xyz_std[iL, :] = np.std(vel_rmse_xyz_runs, axis=0)
        vel_rmse_norm_mean[iL] = np.mean(vel_rmse_norm_runs)
        vel_rmse_norm_std[iL] = np.std(vel_rmse_norm_runs)

        nees_mean[iL, :] = np.mean(nees_runs, axis=0)
        nees_std[iL, :] = np.std(nees_runs, axis=0)

        timing_mean_ms[iL] = np.mean(timing_runs_ms)
        timing_std_ms[iL] = np.std(timing_runs_ms)

        print(f"Finished L_max = {L_max}")
        print(f"  mean position RMSE norm [m] = {pos_rmse_norm_mean[iL]:.3f}")
        print(f"  mean velocity RMSE norm [m/s] = {vel_rmse_norm_mean[iL]:.6f}")
        print(f"  mean timing per step [ms] = {timing_mean_ms[iL]:.3f}")

    # ============================================================
    # SAVE MONTE CARLO SUMMARY
    # ============================================================

    summary_dict = {
        "L_list": np.array(L_list, dtype=np.int64),
        "t_grid": np.array(t_grid, dtype=np.float64),
        "X_truth": np.array(X_truth, dtype=np.float64),

        "pos_rmse_xyz_mean": pos_rmse_xyz_mean,
        "pos_rmse_xyz_std": pos_rmse_xyz_std,
        "pos_rmse_norm_mean": pos_rmse_norm_mean,
        "pos_rmse_norm_std": pos_rmse_norm_std,

        "vel_rmse_xyz_mean": vel_rmse_xyz_mean,
        "vel_rmse_xyz_std": vel_rmse_xyz_std,
        "vel_rmse_norm_mean": vel_rmse_norm_mean,
        "vel_rmse_norm_std": vel_rmse_norm_std,

        "nees_mean": nees_mean,
        "nees_std": nees_std,

        "timing_mean_ms": timing_mean_ms,
        "timing_std_ms": timing_std_ms,
    }

    meta = {
        "run_name": sweep_name,
        "filter_type": "EKF",
        "is_monte_carlo": True,
        "date_str": date_str,
        "seed": int(seed),
        "N_mc": int(N_mc),
        "alt_km": float(alt_km),
        "prop_duration_periods": float(prop_duration),
        "dt_s": float(dt),
        "Ngs": int(len(gs_locations)),
        "sigma_rho": float(sigma_rho),
        "sigma_rhodot": float(sigma_rhodot),
        "elev_mask_deg": float(elev_mask),
        "L_truth": int(L_truth),
        "L_list": L_list,
        "Q_diag": np.diag(Q).tolist(),
        "P0_diag": np.diag(P0).tolist(),
    }

    summary_path = save_mc_summary(
        date_str=date_str,
        sweep_name=sweep_name,
        summary_dict=summary_dict,
        meta=meta
    )

    print("\n============================================================")
    print("Monte Carlo sweep finished.")
    print("Saved summary:", summary_path)
    print("============================================================")


if __name__ == "__main__":
    main()