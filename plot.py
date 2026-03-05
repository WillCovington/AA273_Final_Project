import numpy as np
import matplotlib.pyplot as plt

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
    

def plot_3d_trajectory(ts, X_truth, Xhat, equal_aspect=True, show_projections=False):
    # plots our estimates and the truth trajectory in 3D for funsies
    X_truth = np.asarray(X_truth)
    Xhat = np.asarray(Xhat)

    rT = X_truth[:, :3]
    rH = Xhat[:, :3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(rT[:, 0], rT[:, 1], rT[:, 2], label="Truth")
    ax.plot(rH[:, 0], rH[:, 1], rH[:, 2], label="Estimate")

    ax.scatter(rT[0, 0], rT[0, 1], rT[0, 2], marker="o", s=40, label="Truth start")
    ax.scatter(rH[0, 0], rH[0, 1], rH[0, 2], marker="^", s=40, label="Est start")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Trajectory (Truth vs EKF Estimate)")
    ax.legend(loc="best")

    if equal_aspect:
        # Make axes roughly equal scale (matplotlib doesn't have a perfect built-in for 3D)
        mins = np.minimum(rT.min(axis=0), rH.min(axis=0))
        maxs = np.maximum(rT.max(axis=0), rH.max(axis=0))
        centers = 0.5 * (mins + maxs)
        spans = maxs - mins
        span = float(np.max(spans))
        ax.set_xlim(centers[0] - span/2, centers[0] + span/2)
        ax.set_ylim(centers[1] - span/2, centers[1] + span/2)
        ax.set_zlim(centers[2] - span/2, centers[2] + span/2)

    plt.tight_layout()
    plt.show()

    if show_projections:
        fig2, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(rT[:, 0], rT[:, 1], label="Truth")
        axs[0].plot(rH[:, 0], rH[:, 1], label="Estimate")
        axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]"); axs[0].set_title("XY")
        axs[0].grid(True)

        axs[1].plot(rT[:, 0], rT[:, 2], label="Truth")
        axs[1].plot(rH[:, 0], rH[:, 2], label="Estimate")
        axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("z [m]"); axs[1].set_title("XZ")
        axs[1].grid(True)

        axs[2].plot(rT[:, 1], rT[:, 2], label="Truth")
        axs[2].plot(rH[:, 1], rH[:, 2], label="Estimate")
        axs[2].set_xlabel("y [m]"); axs[2].set_ylabel("z [m]"); axs[2].set_title("YZ")
        axs[2].grid(True)

        axs[0].legend(loc="best")
        plt.tight_layout()
        plt.show()