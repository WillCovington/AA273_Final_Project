import numpy as np
import matplotlib.pyplot as plt
from ground_stations.ground_stations import *
from pathlib import Path
from matplotlib.collections import LineCollection
from gnss.earth_moon_ephemeris import earth_state_mci
from gnss.gnss_satellites import gnss_state_inertial

# we're gonna be doing a lot of plotting here, so I tried to include everything i cou=

def plot_truth_vs_est(ts, X_truth, Xhat, Phat, show_error=True, save_dir=None):
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
    
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(Path(save_dir) / "truth_vs_est.png", dpi=200)

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
        
        if save_dir is not None:
            fig2.savefig(Path(save_dir) / "estimation_error.png", dpi=200)
    
    if save_dir is None:
        plt.show()
    else:
        plt.close(fig1)
        if show_error:
            plt.close(fig2)

def plot_trajectory_with_moon(X_truth, Xhat, model, moon_alpha=0.35, save_dir=None):

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
    
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "3d_orbit.png", dpi=200)
    
    if save_dir is None:
        plt.show(fig)
    else:
        plt.close(fig)
        
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

def _add_colormapped_track(ax, lats, lons, cmap="viridis", linewidth=2.0, label=None):
    """
    Add a single track segment to ax using LineCollection.
    Color varies smoothly from start to end of the segment.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    if len(lats) < 2:
        return None

    points = np.column_stack((lons, lats))
    segments = np.stack([points[:-1], points[1:]], axis=1)

    # color parameter runs from 0 -> 1 across segments
    cvals = np.linspace(0.0, 1.0, len(segments))

    lc = LineCollection(
        segments,
        cmap=cmap,
        linewidths=linewidth,
    )
    lc.set_array(cvals)
    ax.add_collection(lc)

    # fake handle for legend
    if label is not None:
        ax.plot([], [], color=plt.get_cmap(cmap)(0.8), linewidth=linewidth, label=label)

    return lc

def plot_ground_track(
    ts,
    X_truth,
    Xhat=None,
    gs_locations=None,
    img_path="Misc. Notes and Pictures/lroc_color_2k.jpg",
    title="Ground Track (Moon-fixed lat/lon)",
    save_dir=None,
    truth_cmap="viridis",
    est_cmap="plasma",
    linewidth=2.0,
    show_colorbar=False,
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

    # --- Background image ---
    img = plt.imread(img_path)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(img, extent=[-180, 180, -90, 90], origin="upper")

    last_lc_truth = None
    for i, (lats_seg, lons_seg) in enumerate(segs_T):
        lc = _add_colormapped_track(
            ax,
            lats_seg,
            lons_seg,
            cmap=truth_cmap,
            linewidth=linewidth,
            label="Truth" if i == 0 else None,
        )
        if lc is not None:
            last_lc_truth = lc

    last_lc_est = None
    if segs_H is not None:
        for i, (lats_seg, lons_seg) in enumerate(segs_H):
            lc = _add_colormapped_track(
                ax,
                lats_seg,
                lons_seg,
                cmap=est_cmap,
                linewidth=linewidth,
                label="Estimate" if i == 0 else None,
            )
            if lc is not None:
                last_lc_est = lc

    # --- Ground stations ---
    if gs_locations is not None:
        gs_lats = [lat for lat, lon in gs_locations]
        gs_lons = [lon if lon <= 180 else lon - 360 for lat, lon in gs_locations]

        ax.scatter(
            gs_lons,
            gs_lats,
            s=55,
            marker="^",
            color="red",
            edgecolors="black",
            linewidths=0.6,
            label="Ground Stations",
            zorder=5,
        )

    # mark latest positions
    ax.scatter(lon_T[-1], lat_T[-1], s=50, color="white", edgecolors="black", zorder=6)
    if Xhat is not None:
        ax.scatter(lon_H[-1], lat_H[-1], s=50, color="yellow", edgecolors="black", zorder=6)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if show_colorbar and last_lc_truth is not None:
        cbar = fig.colorbar(last_lc_truth, ax=ax)
        cbar.set_label("Track progression")

    plt.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "ground_track.png", dpi=200)
        plt.close(fig)
    else:
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
    
# the next fun, dumb thing to plot is the entire earth-moon gnss and lunar satellite system

R_EARTH_M = 6378.1363e3


def _make_sphere(radius, center, nu=50, nv=25):
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    return x, y, z

def plot_earth_moon_gnss_dual_view(
    ts,
    X_truth,
    model,
    gnss_sats,
    Xhat=None,
    t_plot=None,
    moon_alpha=0.35,
    earth_alpha=0.35,
    save_dir=None,
    moon_orbit_scale_system=20.0,
):
    # because I was a dumbass and modeled everything in lunar coordinates rather than ECI ones, we have to do some conversions later on
    # all of the inputs make sense like the other plot calls, but here are some notable inputs  
    # gnss_sats: our list of gnss_satellites (shocker)
    # t_plot: optional input to indicate if we want to draw the satellites at a different instant in time
    # Xhat: optional input if we want to include our estimates
    # zoom_mode: because we're dealing with 2 very far away bodies, we can either show the full scale (but actually scaled by moon_orbit_scale) or we can just show the moon
    # moon_orbit_scale: makes the system zoom mode a bit more tolerable by reducing the distance between the bodies for ease of visibility. use 1.0 for true scale

    ts = np.asarray(ts, dtype=np.float64).reshape(-1,)
    X_truth = np.asarray(X_truth, dtype=np.float64)

    if t_plot is None:
        t_plot = float(ts[-1])

    # Truth and filtered trajectories
    rT_mci = X_truth[:, :3].copy()
    rH_mci = None
    if Xhat is not None:
        Xhat = np.asarray(Xhat, dtype=np.float64)
        rH_mci = Xhat[:, :3].copy()

    # Earth and GNSS states at chosen time (probably just gonna be the end of runtime)
    r_earth_mci, _ = earth_state_mci(t_plot)

    gnss_pos_mci = []
    for sat in gnss_sats:
        r_gnss, _ = gnss_state_inertial(sat, t_plot)
        gnss_pos_mci.append(r_gnss)
    gnss_pos_mci = np.asarray(gnss_pos_mci, dtype=np.float64)

    fig = plt.figure(figsize=(15, 7))

    # Left Panel: Earth-centered system view
    ax1 = fig.add_subplot(121, projection="3d")

    # Shift into Earth-centered frame
    rT_ec = rT_mci - r_earth_mci
    rT_ec_plot = rT_ec * moon_orbit_scale_system

    rH_ec_plot = None
    if rH_mci is not None:
        rH_ec = rH_mci - r_earth_mci
        rH_ec_plot = rH_ec * moon_orbit_scale_system

    moon_center_ec = -r_earth_mci
    gnss_pos_ec = gnss_pos_mci - r_earth_mci
    earth_center_ec = np.zeros(3)

    # Earth
    x_e, y_e, z_e = _make_sphere(R_EARTH_M, earth_center_ec)
    ax1.plot_surface(x_e, y_e, z_e, color="blue", alpha=earth_alpha, linewidth=0)

    # Moon
    x_m, y_m, z_m = _make_sphere(model.r0_m, moon_center_ec)
    ax1.plot_surface(x_m, y_m, z_m, color="gray", alpha=moon_alpha, linewidth=0)

    # GNSS satellites
    ax1.scatter(
        gnss_pos_ec[:, 0], gnss_pos_ec[:, 1], gnss_pos_ec[:, 2],
        s=18, label="GNSS satellites"
    )

    # Moon orbiter trajectory (scaled for visibility)
    ax1.plot(
        rT_ec_plot[:, 0], rT_ec_plot[:, 1], rT_ec_plot[:, 2],
        linewidth=2, label="Moon satellite truth (scaled)"
    )
    ax1.scatter(
        rT_ec_plot[-1, 0], rT_ec_plot[-1, 1], rT_ec_plot[-1, 2],
        s=35, label="Truth final"
    )

    if rH_ec_plot is not None:
        ax1.plot(
            rH_ec_plot[:, 0], rH_ec_plot[:, 1], rH_ec_plot[:, 2],
            linewidth=2, linestyle="--", label="Moon satellite est. (scaled)"
        )
        ax1.scatter(
            rH_ec_plot[-1, 0], rH_ec_plot[-1, 1], rH_ec_plot[-1, 2],
            s=35, label="Estimate final"
        )

    # Mark Earth and Moon centers
    ax1.scatter(0.0, 0.0, 0.0, s=60, marker="x", label="Earth center")
    ax1.scatter(moon_center_ec[0], moon_center_ec[1], moon_center_ec[2], s=60, marker="x", label="Moon center")

    ax1.set_title("Earth-Centered System View")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")

    all_points_left = [gnss_pos_ec, np.array([moon_center_ec]), np.array([[0.0, 0.0, 0.0]])]
    all_points_left.append(np.array([
        moon_center_ec + np.array([ model.r0_m, 0.0, 0.0]),
        moon_center_ec + np.array([-model.r0_m, 0.0, 0.0]),
        np.array([ R_EARTH_M, 0.0, 0.0]),
        np.array([-R_EARTH_M, 0.0, 0.0]),
    ]))
    all_points_left = np.vstack(all_points_left)

    ax1.set_xlim(all_points_left[:, 0].min(), all_points_left[:, 0].max())
    ax1.set_ylim(all_points_left[:, 1].min(), all_points_left[:, 1].max())
    ax1.set_zlim(all_points_left[:, 2].min(), all_points_left[:, 2].max())
    set_axes_equal(ax1)
    ax1.legend(loc="best", fontsize=8)

    # Right Panel: Moon Centered Plot
    ax2 = fig.add_subplot(122, projection="3d")

    # Moon
    x_m2, y_m2, z_m2 = _make_sphere(model.r0_m, np.zeros(3))
    ax2.plot_surface(x_m2, y_m2, z_m2, color="gray", alpha=moon_alpha, linewidth=0)

    # Local orbit truth
    ax2.plot(
        rT_mci[:, 0], rT_mci[:, 1], rT_mci[:, 2],
        linewidth=2, label="Moon satellite truth"
    )
    ax2.scatter(
        rT_mci[-1, 0], rT_mci[-1, 1], rT_mci[-1, 2],
        s=35, label="Truth final"
    )

    if rH_mci is not None:
        ax2.plot(
            rH_mci[:, 0], rH_mci[:, 1], rH_mci[:, 2],
            linewidth=2, linestyle="--", label="Moon satellite estimate"
        )
        ax2.scatter(
            rH_mci[-1, 0], rH_mci[-1, 1], rH_mci[-1, 2],
            s=35, label="Estimate final"
        )

    ax2.set_title("Moon-Centered Local Orbit View")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")

    all_points_right = [rT_mci]
    if rH_mci is not None:
        all_points_right.append(rH_mci)
    all_points_right.append(np.array([
        [ model.r0_m, 0.0, 0.0],
        [-model.r0_m, 0.0, 0.0],
        [0.0, model.r0_m, 0.0],
        [0.0, -model.r0_m, 0.0],
        [0.0, 0.0, model.r0_m],
        [0.0, 0.0, -model.r0_m],
    ]))
    all_points_right = np.vstack(all_points_right)

    ax2.set_xlim(all_points_right[:, 0].min(), all_points_right[:, 0].max())
    ax2.set_ylim(all_points_right[:, 1].min(), all_points_right[:, 1].max())
    ax2.set_zlim(all_points_right[:, 2].min(), all_points_right[:, 2].max())
    set_axes_equal(ax2)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"Earth-Moon-GNSS Geometry at t = {t_plot:.1f} s", fontsize=14)
    fig.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "earth_moon_gnss_dual_view.png", dpi=220)
        plt.close(fig)
    else:
        plt.show()