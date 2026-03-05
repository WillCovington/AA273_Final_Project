import numpy as np
import matplotlib.pyplot as plt
from ground_stations import *

# we're gonna be doing a lot of plotting here, so I tried to include everything i cou=

def plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error=True):
    # takes in our times, true state, estimated state, and covariance estimates and returns the 

    ts = np.asarray(ts).reshape(-1,)
    X_truth = np.asarray(X_truth)
    Xhat = np.asarray(Xhat)

    # checks if our state arrays are of equal size
    if X_truth.shape != Xhat.shape:
        raise ValueError(f"X_truth shape {X_truth.shape} must match Xhat shape {Xhat.shape}")

    # checks 
    K, n = Xhat.shape
    if n != 6:
        raise ValueError("Expected 6-state vectors (x,y,z,vx,vy,vz)")

    if len(Phat) != K:
        raise ValueError(f"Phat length {len(Phat)} must equal K={K}")

    # this is how many standard deviations we want to show in our confidence interval
    # 95% is roughly 2 sigma, so we just set it to 2
    k_sigma = 2

    # Extract 1-sigma for each state over time from covariance diagonals
    sig = np.zeros((K, 6), dtype=float)
    for i in range(K):
        P = np.asarray(Phat[i])
        sig[i, :] = np.sqrt(np.maximum(np.diag(P), 0.0))

    labels = ["x [m]", "y [m]", "z [m]", "vx [m/s]", "vy [m/s]", "vz [m/s]"]

    # Truth vs. Estimate
    fig1, axes1 = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes1 = axes1.ravel()

    for j in range(6):
        ax = axes1[j]
        ax.plot(ts, X_truth[:, j], label="Truth")
        ax.plot(ts, Xhat[:, j], label="Estimate")

        upper = Xhat[:, j] + k_sigma * sig[:, j]
        lower = Xhat[:, j] - k_sigma * sig[:, j]
        ax.fill_between(ts, lower, upper, alpha=0.2, label="95% CI" if j == 0 else None)

        ax.set_ylabel(labels[j])
        ax.grid(True)

    axes1[-2].set_xlabel("Time [s]")
    axes1[-1].set_xlabel("Time [s]")
    axes1[0].legend(loc="best")
    fig1.suptitle("EKF: Truth vs Estimate with 95% Confidence Bounds", y=0.98)
    fig1.tight_layout()

    # Just the estimation error (estimate minus truth)
    if show_error:
        err = Xhat - X_truth

        fig2, axes2 = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        axes2 = axes2.ravel()

        for j in range(6):
            ax = axes2[j]
            ax.plot(ts, err[:, j], label="Error (est - truth)")

            upper = +k_sigma * sig[:, j]
            lower = -k_sigma * sig[:, j]
            ax.fill_between(ts, lower, upper, alpha=0.2, label="±95% CI" if j == 0 else None)
            ax.axhline(0.0, linewidth=1)

            ax.set_ylabel(labels[j])
            ax.grid(True)

        axes2[-2].set_xlabel("Time [s]")
        axes2[-1].set_xlabel("Time [s]")
        axes2[0].legend(loc="best")
        fig2.suptitle("EKF: Estimation Error with ±95% Bounds", y=0.98)
        fig2.tight_layout()

    plt.show()

def plot_trajectory_with_moon(X_truth, Xhat, model, moon_alpha=0.35):

    rT = X_truth[:, :3]
    rH = Xhat[:, :3]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # plot our actual trajectories
    ax.plot(rT[:,0], rT[:,1], rT[:,2], label="Truth", linewidth=2)
    ax.plot(rH[:,0], rH[:,1], rH[:,2], label="Estimate", linewidth=2, linestyle='--')

    # set up the moon time
    r_moon = model.r0_m

    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)

    x = r_moon * np.outer(np.cos(u), np.sin(v))
    y = r_moon * np.outer(np.sin(u), np.sin(v))
    z = r_moon * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color="gray", alpha=moon_alpha, linewidth=0)

    # ---- Labels ----
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Satellite Orbit Around Moon")

    ax.legend()

    all_points = np.vstack((rT, rH, np.array([
        [ r_moon,0,0],[-r_moon,0,0],
        [0, r_moon,0],[0,-r_moon,0],
        [0,0, r_moon],[0,0,-r_moon]
    ])))

    ax.set_xlim(all_points[:,0].min(), all_points[:,0].max())
    ax.set_ylim(all_points[:,1].min(), all_points[:,1].max())
    ax.set_zlim(all_points[:,2].min(), all_points[:,2].max())

    # set axes equal
    set_axes_equal(ax)

    plt.tight_layout()
    plt.show()
    
# Stuff for plotting the ground track
def _split_dateline(lats, lons, jump_deg=180.0):
    lats = np.asarray(lats); lons = np.asarray(lons)
    segs = []
    start = 0
    for k in range(1, len(lons)):
        if np.abs(lons[k] - lons[k-1]) > jump_deg:
            segs.append((lats[start:k], lons[start:k]))
            start = k
    segs.append((lats[start:], lons[start:]))
    return segs

def plot_ground_track(
    ts,
    X_truth,
    Xhat=None,
    img_path="Misc. Notes and Pictures\lroc_color_2k.jpg",
    title="Ground Track (Moon-fixed lat/lon)",
):
    ts = np.asarray(ts, dtype=np.float64).reshape(-1,)
    X_truth = np.asarray(X_truth, dtype=np.float64)

    # --- Convert truth to lat/lon ---
    lat_T = np.zeros_like(ts)
    lon_T = np.zeros_like(ts)
    for k, t in enumerate(ts):
        lat_T[k], lon_T[k] = inertial_to_latlon(X_truth[k, :3], float(t))

    segs_T = _split_dateline(lat_T, lon_T)

    # --- Convert estimate if provided ---
    segs_H = None
    if Xhat is not None:
        Xhat = np.asarray(Xhat, dtype=np.float64)
        lat_H = np.zeros_like(ts)
        lon_H = np.zeros_like(ts)
        for k, t in enumerate(ts):
            lat_H[k], lon_H[k] = inertial_to_latlon(Xhat[k, :3], float(t))
        segs_H = _split_dateline(lat_H, lon_H)

    # --- Plot background image ---
    img = plt.imread(img_path)

    fig, ax = plt.subplots(figsize=(14, 6))
    # Extent corresponds to lon [-180, 180], lat [-90, 90]
    ax.imshow(img, extent=[-180, 180, -90, 90], origin="upper")

    # --- Plot segments (truth) ---
    for lats_seg, lons_seg in segs_T:
        ax.plot(lons_seg, lats_seg, linewidth=2, label="Truth" if lats_seg is segs_T[0][0] else None)

    # --- Plot estimate segments ---
    if segs_H is not None:
        for lats_seg, lons_seg in segs_H:
            ax.plot(lons_seg, lats_seg, linewidth=2, linestyle="--",
                    label="Estimate" if lats_seg is segs_H[0][0] else None)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
def set_axes_equal(ax):
    # I'll be honest this helper function is from Chat
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])