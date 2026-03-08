import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_runs(date_dir):
    """
    Load all saved run npz files from runs/<date_dir>/<runname>/<runname>.npz
    """
    date_dir = Path(date_dir)
    runs = []

    for run_dir in sorted(date_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        npz_files = list(run_dir.glob("*.npz"))
        if not npz_files:
            continue

        data = np.load(npz_files[0], allow_pickle=True)
        meta = json.loads(str(data["meta_json"]))

        runs.append({
            "run_dir": run_dir,
            "L_max": int(meta["L_max"]),
            "meta": meta,
            "ts": data["ts"],
            "X_truth": data["X_truth"],
            "Xhat": data["Xhat"],
            "Phat": data["Phat"],
            "timing_total": data["timing_total_step_s"] if "timing_total_step_s" in data else None,
            "timing_prop": data["timing_propagate_s"] if "timing_propagate_s" in data else None,
            "timing_meas": data["timing_meas_model_s"] if "timing_meas_model_s" in data else None,
            "timing_update": data["timing_update_s"] if "timing_update_s" in data else None,
        })

    runs.sort(key=lambda r: r["L_max"])
    return runs


def compute_rmse(X_truth, Xhat):
    err = Xhat - X_truth

    pos_err_vec = err[:, :3]
    vel_err_vec = err[:, 3:]

    pos_err_norm = np.linalg.norm(pos_err_vec, axis=1)
    vel_err_norm = np.linalg.norm(vel_err_vec, axis=1)

    rmse_pos = float(np.sqrt(np.mean(pos_err_norm**2)))
    rmse_vel = float(np.sqrt(np.mean(vel_err_norm**2)))

    return rmse_pos, rmse_vel, pos_err_norm, vel_err_norm


def compute_nees(X_truth, Xhat, Phat):
    K = X_truth.shape[0]
    nees = np.zeros(K, dtype=np.float64)

    for k in range(K):
        e = (Xhat[k] - X_truth[k]).reshape(-1, 1)
        P = Phat[k]
        nees[k] = float(e.T @ np.linalg.solve(P, e))

    return nees


def make_sweep_report(
    date_dir,
    save_path=None,
    show=True,
    pos_requirement_m=10.0,
    runtime_logscale=False,
):
    runs = load_runs(date_dir)

    if not runs:
        raise ValueError(f"No runs found in {date_dir}")

    Ls = []
    rmse_pos = []
    rmse_vel = []
    nees_mean = []

    runtime_total = []
    runtime_prop = []
    runtime_meas = []
    runtime_update = []

    frac_below_req = []
    final_pos_err = []

    for r in runs:
        Ls.append(r["L_max"])

        rp, rv, pos_err_norm, vel_err_norm = compute_rmse(r["X_truth"], r["Xhat"])
        rmse_pos.append(rp)
        rmse_vel.append(rv)

        nees = compute_nees(r["X_truth"], r["Xhat"], r["Phat"])
        nees_mean.append(float(np.mean(nees)))

        frac_below_req.append(float(np.mean(pos_err_norm <= pos_requirement_m)))
        final_pos_err.append(float(pos_err_norm[-1]))

        if r["timing_total"] is not None:
            runtime_total.append(1e3 * float(np.mean(r["timing_total"])))
        else:
            runtime_total.append(np.nan)

        if r["timing_prop"] is not None:
            runtime_prop.append(1e3 * float(np.mean(r["timing_prop"])))
        else:
            runtime_prop.append(np.nan)

        if r["timing_meas"] is not None:
            runtime_meas.append(1e3 * float(np.mean(r["timing_meas"])))
        else:
            runtime_meas.append(np.nan)

        if r["timing_update"] is not None:
            runtime_update.append(1e3 * float(np.mean(r["timing_update"])))
        else:
            runtime_update.append(np.nan)

    Ls = np.asarray(Ls)
    rmse_pos = np.asarray(rmse_pos)
    rmse_vel = np.asarray(rmse_vel)
    nees_mean = np.asarray(nees_mean)

    runtime_total = np.asarray(runtime_total)
    runtime_prop = np.asarray(runtime_prop)
    runtime_meas = np.asarray(runtime_meas)
    runtime_update = np.asarray(runtime_update)

    frac_below_req = np.asarray(frac_below_req)
    final_pos_err = np.asarray(final_pos_err)

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.ravel()

    # ---------------- Panel 1: RMSE ----------------
    ax = axs[0]
    ax.plot(Ls, rmse_pos, marker="o", label="Position RMSE")
    ax.plot(Ls, rmse_vel, marker="s", label="Velocity RMSE")
    ax.set_title("Accuracy vs Truncation Degree")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("RMSE")
    ax.grid(True)
    ax.legend()

    # ---------------- Panel 2: NEES ----------------
    ax = axs[1]
    ax.plot(Ls, nees_mean, marker="o", label="Mean NEES")
    ax.axhline(6.0, linestyle="--", label="Expected NEES = 6")
    ax.set_title("Consistency vs Truncation Degree")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("Mean NEES")
    ax.grid(True)
    ax.legend()

    # ---------------- Panel 3: Total runtime ----------------
    ax = axs[2]
    ax.plot(Ls, runtime_total, marker="o")
    if runtime_logscale:
        ax.set_yscale("log")
    ax.set_title("Runtime vs Truncation Degree")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("Mean step time [ms]")
    ax.grid(True)

    # ---------------- Panel 4: Runtime breakdown ----------------
    ax = axs[3]
    ax.plot(Ls, runtime_prop, marker="o", label="Propagation")
    ax.plot(Ls, runtime_meas, marker="o", label="Measurement model")
    ax.plot(Ls, runtime_update, marker="o", label="Update")
    if runtime_logscale:
        ax.set_yscale("log")
    ax.set_title("Runtime Breakdown")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("Mean step time [ms]")
    ax.grid(True)
    ax.legend()

    # ---------------- Panel 5: Fraction meeting requirement ----------------
    ax = axs[4]
    ax.plot(Ls, frac_below_req, marker="o")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Fraction of Time Position Error < {pos_requirement_m:g} m")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("Fraction of run")
    ax.grid(True)

    # ---------------- Panel 6: Final position error ----------------
    ax = axs[5]
    ax.plot(Ls, final_pos_err, marker="o")
    ax.axhline(pos_requirement_m, linestyle="--", label=f"{pos_requirement_m:g} m requirement")
    ax.set_title("Final Position Error vs Truncation Degree")
    ax.set_xlabel(r"$L_{max}$")
    ax.set_ylabel("Final position error [m]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Filter Sweep Summary", fontsize=15, y=0.98)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250)

    if show:
        plt.show()
    else:
        plt.close(fig)