"""
analysis/plot_mc_results.py

- loads EKF and UKF Monte Carlo summary .npz files
- makes comparison plots for the gravity fidelity sweep results

Plots:
1) Position RMSE vs L
2) Mean NEES vs L
3) NEES vs time for some L values
4) Velocity RMSE vs L
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================
EKF_SUMMARY_PATH = "runs/03-12-2026/ekf_mc_sweep_30km/ekf_mc_sweep_30km_mc_summary.npz"
UKF_SUMMARY_PATH = "runs/03-12-2026/ukf_mc_sweep_30km/ukf_mc_sweep_30km_mc_summary.npz"

# Where to save the figures
OUT_DIR = "runs/03-12-2026-30km/30km_mc_comparison_figures"

# Selected gravity degrees for NEES-vs-time plots
SELECTED_L = [5, 50, 200, 600] # removed 2, 10

# Save format
FIG_DPI = 300
SAVE_PNG = True
SHOW_FIGS = True

# If True, I use log x-axis for L
USE_LOG_X = False


# ============================================================
# DATA HELPERS
# ============================================================

def load_summary(npz_path: str) -> dict:
    """
    Load one Monte Carlo summary .npz into a plain dict.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find summary file: {npz_path}")

    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out


def get_L_indices(L_list: np.ndarray, selected_L: list[int]) -> list[tuple[int, int]]:
    """
    Return [(idx, L), ...] only for L values that exist in L_list.
    """
    pairs = []
    for L in selected_L:
        matches = np.where(L_list == L)[0]
        if len(matches) > 0:
            pairs.append((int(matches[0]), int(L)))
    return pairs


def mean_nees_over_time(nees_mean: np.ndarray) -> np.ndarray:
    """
    transform NEES time history into one mean value per L.
    Input shape: (nL, K)
    Output shape: (nL,)
    """
    return np.mean(nees_mean, axis=1)


def final_nees(nees_mean: np.ndarray) -> np.ndarray:
    """
    Final-time NEES for each L.
    Input shape: (nL, K)
    Output shape: (nL,)
    """
    return nees_mean[:, -1]


def maybe_log_x(ax):
    if USE_LOG_X:
        ax.set_xscale("log")


def save_fig(fig, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(out_dir / filename, dpi=FIG_DPI, bbox_inches="tight")


# ============================================================
# PLOTS
# ============================================================

def plot_position_rmse_vs_L(ekf: dict, ukf: dict, out_dir: Path):
    """
    EKF vs UKF: position RMSE vs gravity degree
    """
    L_ekf = ekf["L_list"]
    L_ukf = ukf["L_list"]

    y_ekf = ekf["pos_rmse_norm_mean"]
    y_ukf = ukf["pos_rmse_norm_mean"]

    e_ekf = ekf["pos_rmse_norm_std"]
    e_ukf = ukf["pos_rmse_norm_std"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(L_ekf, y_ekf, yerr=e_ekf, marker="o", linestyle="-", capsize=3, label="EKF")
    ax.errorbar(L_ukf, y_ukf, yerr=e_ukf, marker="s", linestyle="-", capsize=3, label="UKF")

    maybe_log_x(ax)
    ax.set_xlabel("Gravity truncation degree $L_{max}$")
    ax.set_ylabel("Position RMSE [m]")
    ax.set_title("Position RMSE vs Gravity Model Fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(fig, out_dir, "position_rmse_vs_L.png")
    return fig


def plot_velocity_rmse_vs_L(ekf: dict, ukf: dict, out_dir: Path):
    """
    EKF vs UKF: velocity RMSE vs gravity degree
    """
    L_ekf = ekf["L_list"]
    L_ukf = ukf["L_list"]

    y_ekf = ekf["vel_rmse_norm_mean"]
    y_ukf = ukf["vel_rmse_norm_mean"]

    e_ekf = ekf["vel_rmse_norm_std"]
    e_ukf = ukf["vel_rmse_norm_std"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(L_ekf, y_ekf, yerr=e_ekf, marker="o", linestyle="-", capsize=3, label="EKF")
    ax.errorbar(L_ukf, y_ukf, yerr=e_ukf, marker="s", linestyle="-", capsize=3, label="UKF")

    maybe_log_x(ax)
    ax.set_xlabel("Gravity truncation degree $L_{max}$")
    ax.set_ylabel("Velocity RMSE [m/s]")
    ax.set_title("Velocity RMSE vs Gravity Model Fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(fig, out_dir, "velocity_rmse_vs_L.png")
    return fig


def plot_mean_nees_vs_L(ekf: dict, ukf: dict, out_dir: Path):
    """
    EKF vs UKF: time-averaged NEES vs gravity degree
    """
    L_ekf = ekf["L_list"]
    L_ukf = ukf["L_list"]

    y_ekf = mean_nees_over_time(ekf["nees_mean"])
    y_ukf = mean_nees_over_time(ukf["nees_mean"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(L_ekf, y_ekf, marker="o", linestyle="-", label="EKF")
    ax.plot(L_ukf, y_ukf, marker="s", linestyle="-", label="UKF")

    maybe_log_x(ax)
    ax.set_xlabel("Gravity truncation degree $L_{max}$")
    ax.set_ylabel("Mean NEES")
    ax.set_title("Mean NEES vs Gravity Model Fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(fig, out_dir, "mean_nees_vs_L.png")
    return fig


def plot_final_nees_vs_L(ekf: dict, ukf: dict, out_dir: Path):
    """
    EKF vs UKF: final-time NEES vs gravity degree
    """
    L_ekf = ekf["L_list"]
    L_ukf = ukf["L_list"]

    y_ekf = final_nees(ekf["nees_mean"])
    y_ukf = final_nees(ukf["nees_mean"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(L_ekf, y_ekf, marker="o", linestyle="-", label="EKF")
    ax.plot(L_ukf, y_ukf, marker="s", linestyle="-", label="UKF")

    maybe_log_x(ax)
    ax.set_xlabel("Gravity truncation degree $L_{max}$")
    ax.set_ylabel("Final-time NEES")
    ax.set_title("Final-Time NEES vs Gravity Model Fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(fig, out_dir, "final_nees_vs_L.png")
    return fig


def plot_nees_vs_time_selected_L(summary: dict, filter_name: str, selected_L: list[int], out_dir: Path):
    """
    One figure for one filter: NEES vs time for selected L values
    """
    L_list = summary["L_list"]
    t_grid = summary["t_grid"]
    nees = summary["nees_mean"]

    pairs = get_L_indices(L_list, selected_L)
    if len(pairs) == 0:
        raise ValueError(f"No selected L values found in {filter_name} summary.")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, L in pairs:
        ax.plot(t_grid, nees[idx, :], label=f"L={L}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean NEES")
    ax.set_title(f"{filter_name}: NEES vs Time for Selected Gravity Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fname = f"{filter_name.lower()}_nees_vs_time_selected_L.png"
    save_fig(fig, out_dir, fname)
    return fig


def plot_nees_vs_time_overlay(ekf: dict, ukf: dict, selected_L: list[int], out_dir: Path):
    """
    One figure per selected L: EKF vs UKF NEES vs time
    """
    L_ekf = ekf["L_list"]
    L_ukf = ukf["L_list"]

    t_ekf = ekf["t_grid"]
    t_ukf = ukf["t_grid"]

    # Only plot L values present in both
    common_L = [L for L in selected_L if (L in L_ekf) and (L in L_ukf)]

    figs = []

    for L in common_L:
        idx_ekf = int(np.where(L_ekf == L)[0][0])
        idx_ukf = int(np.where(L_ukf == L)[0][0])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(t_ekf, ekf["nees_mean"][idx_ekf, :], linestyle="-", label=f"EKF, L={L}")
        ax.plot(t_ukf, ukf["nees_mean"][idx_ukf, :], linestyle="--", label=f"UKF, L={L}")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean NEES")
        ax.set_title(f"NEES vs Time Comparison at L={L}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        save_fig(fig, out_dir, f"nees_vs_time_overlay_L{L}.png")
        figs.append(fig)

    return figs


# ============================================================
# MAIN
# ============================================================

def main():
    out_dir = Path(OUT_DIR)

    ekf = load_summary(EKF_SUMMARY_PATH)
    ukf = load_summary(UKF_SUMMARY_PATH)

    # Basic checks
    required_keys = [
        "L_list",
        "t_grid",
        "pos_rmse_norm_mean",
        "pos_rmse_norm_std",
        "vel_rmse_norm_mean",
        "vel_rmse_norm_std",
        "nees_mean",
    ]

    for key in required_keys:
        if key not in ekf:
            raise KeyError(f"Missing key '{key}' in EKF summary.")
        if key not in ukf:
            raise KeyError(f"Missing key '{key}' in UKF summary.")

    print("Loaded EKF summary from:", EKF_SUMMARY_PATH)
    print("Loaded UKF summary from:", UKF_SUMMARY_PATH)
    print("Saving figures to:", out_dir)

    figs = []
    figs.append(plot_position_rmse_vs_L(ekf, ukf, out_dir))
    figs.append(plot_velocity_rmse_vs_L(ekf, ukf, out_dir))
    figs.append(plot_mean_nees_vs_L(ekf, ukf, out_dir))
    figs.append(plot_final_nees_vs_L(ekf, ukf, out_dir))
    figs.append(plot_nees_vs_time_selected_L(ekf, "EKF", SELECTED_L, out_dir))
    figs.append(plot_nees_vs_time_selected_L(ukf, "UKF", SELECTED_L, out_dir))

    overlay_figs = plot_nees_vs_time_overlay(ekf, ukf, SELECTED_L, out_dir)
    figs.extend(overlay_figs)

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()