import numpy as np
import json

from metrics import position_rmse, velocity_rmse, nees_series
from plot import plot_truth_vs_est, plot_trajectory_with_moon, plot_ground_track

def load_run(path):
    data = np.load(path, allow_pickle=True)
    ts = data["ts"]
    X_truth = data["X_truth"]
    Xhat = data["Xhat"]
    Phat = data["Phat"]          # (K,6,6)
    meta = json.loads(str(data["meta_json"]))
    return ts, X_truth, Xhat, Phat, meta

def analyze_run(path, model=None, ground_img_path=None, save_dir = None):
    ts, X_truth, Xhat, Phat, meta = load_run(path)

    # --- Metrics ---
    rmse_xyz, rmse_pos = position_rmse(X_truth, Xhat)
    rmse_vxyz, rmse_vel = velocity_rmse(X_truth, Xhat)
    nees = nees_series(X_truth, Xhat, Phat)

    print("\n=== Run:", meta.get("run_name", "unknown"), "===")
    print("RMSE position [m] xyz:", rmse_xyz)
    print("RMSE position norm [m]:", rmse_pos)
    print("RMSE velocity [m/s] xyz:", rmse_vxyz)
    print("RMSE velocity norm [m/s]:", rmse_vel)
    print("Mean NEES:", float(np.mean(nees)))

    # --- Plots you already have ---
    plot_truth_vs_est(ts, X_truth, Xhat, list(Phat), show_error=True)

    if model is not None:
        plot_trajectory_with_moon(X_truth, Xhat, model)

    if ground_img_path is not None:
        plot_ground_track(ts, X_truth, Xhat=Xhat, img_path=ground_img_path)

    return {
        "rmse_pos_xyz": rmse_xyz,
        "rmse_pos_norm": rmse_pos,
        "rmse_vel_xyz": rmse_vxyz,
        "rmse_vel_norm": rmse_vel,
        "nees_mean": float(np.mean(nees)),
        "nees_series": nees,
        "meta": meta,
    }