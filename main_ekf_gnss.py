import numpy as np

from gravity.gravity_model import GravityModel
from gravity.dynamics import make_time_grid, rollout
from analysis.plot import plot_truth_vs_est, plot_trajectory_with_moon, plot_ground_track, plot_earth_moon_gnss_dual_view
from analysis.save_run import *

from gnss.gnss_satellites import define_gnss_constellation
from gnss.gnss_measurements import *
from gnss.ekf_gnss_fun import ekf_run_gnss

from analysis.sweep_report import *


def main():
    date_str = "03-09-2026"   # update as needed
    seed = 269
    rng = np.random.default_rng(seed)

    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")

    ##################### SETTING UP OUR TRUTH MODEL #####################
    alt_km = 50.0 # altitude of orbit in km
    r_mag = model.r0_m + alt_km * 1000.0
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu / r_mag)
    T_period = 2 * np.pi * np.sqrt(r_mag**3 / mu)
    prop_duration = 1.0   # number of periods

    # truth initial state
    # no matter how we want our orbit to work, we'll probably wanna leave the final entry (velocity in the z direction) as v_circ and leave the other velocity entries as zero
    # however, if we want to shift our orbital plane at all, just change the beginning longitude below and it moves our orbit
    long_init = 75 # initial longitude in degrees
    initial_x = r_mag * np.cos(np.deg2rad(long_init))
    initial_y = r_mag * np.sin(np.deg2rad(long_init))
    x0_truth = np.array([initial_x, initial_y, 0.0, 0.0, 0.0, v_circ], dtype=np.float64)

    # time grid (final entry is the rate of propogation update -- this is carried through to the state filter portion)
    update_rate = 5.0
    t_grid = make_time_grid(0.0, T_period * prop_duration, update_rate)

    L_truth = model.lmax_data

    # truth trajectory
    X_truth = rollout(x0_truth, t_grid, L_truth, model)
    print("---------------------------Truth Propagation Finished---------------------------")

    ##################### SETTING UP GNSS MEASUREMENTS #####################

    # Synthetic GNSS constellation
    gnss_sats = define_gnss_constellation(
        n_planes=6,
        sats_per_plane=4,
        sma_m=26560e3,
        inc_deg=55.0,
    )

    sigma_rho = 5.0 # [m]
    sigma_rhodot = 0.05 # [m/s]

    R_full = build_R_full_gnss(len(gnss_sats), sigma_rho, sigma_rhodot)

    # Generate measurements once from truth
    measurements = []
    for k, t in enumerate(t_grid):
        y = take_gnss_measurements(
            X_truth[k],
            gnss_sats,
            float(t),
            sigma_rho=sigma_rho,
            sigma_rhodot=sigma_rhodot,
            add_noise=True,
            rng=rng,
            occultation_margin_m=0.0,
        )
        measurements.append({"t": float(t), "y": y, "gnss": gnss_sats})

    print("---------------------------Truth GNSS Measurements Finished---------------------------")

    ##################### RUNNING THROUGH OUR EKF ESTIMATES AT VARYING L #####################

    x0 = x0_truth.copy()
    P0 = np.diag([1e5, 1e5, 1e5, 1e0, 1e0, 1e0]).astype(np.float64)

    Q = np.eye(6, dtype=np.float64) * 1e-6

    L_list = [2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600]

    for L_max in L_list:
        runname = f"gnss_ekf_L{L_max}_Nsat{len(gnss_sats)}_alt{alt_km}_T{prop_duration}per_seed{seed}"

        print("\n==============================")
        print("Running GNSS EKF with L_max =", L_max)

        ts, Xhat, Phat, timing = ekf_run_gnss(
            x0=x0,
            P0=P0,
            measurements=measurements,
            model=model,
            L_max=L_max,
            Q=Q,
            R_full=R_full,
            eps_fd=5.0,
        )

        meta = {
            "run_name": runname,
            "filter_type": "EKF_GNSS",
            "seed": int(seed),
            "alt_km": float(alt_km),
            "prop_duration_periods": float(prop_duration),
            "dt_s": 1.0,
            "Nsat": int(len(gnss_sats)),
            "sigma_rho": float(sigma_rho),
            "sigma_rhodot": float(sigma_rhodot),
            "L_truth": int(L_truth),
            "L_max": int(L_max),
            "Q_diag": np.diag(Q).tolist(),
        }

        npz_path, fig_dir = save_run(
            date_str,
            runname,
            ts,
            X_truth,
            Xhat,
            Phat,
            meta,
            timing=timing,
        )

        print("Saved:", npz_path)
        print("Timing mean total step [ms]:", 1e3 * float(np.mean(timing["total_step_s"])))

        # Save figures
        plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error=True, save_dir=fig_dir)
        plot_earth_moon_gnss_dual_view(ts=ts, X_truth=X_truth, Xhat=Xhat, model=model, gnss_sats=gnss_sats, t_plot=ts[-1], save_dir=fig_dir, moon_orbit_scale_system=100.0)
        # ground track does not show GNSS sats, but still useful for spacecraft path
        plot_ground_track(
            ts,
            X_truth,
            Xhat,
            img_path="Misc. Notes and Pictures/lroc_color_2k.jpg",
            save_dir=fig_dir,
        )
    make_sweep_report(date_dir= "runs/{date_str}", save_path="runs/{date_str}/sweep_report.png", show=True, pos_requirement_m=10.0)

if __name__ == "__main__":
    main()