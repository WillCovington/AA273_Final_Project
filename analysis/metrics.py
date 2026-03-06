import numpy as np

def rmse(a, b, axis=0):
    # calculates root mean square error
    a = np.asarray(a); b = np.asarray(b)
    return np.sqrt(np.mean((a - b)**2, axis=axis))

def position_rmse(X_truth, Xhat):
    # calculates position rmse
    e = Xhat[:, :3] - X_truth[:, :3]
    rmse_xyz = np.sqrt(np.mean(e**2, axis=0))
    rmse_norm = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    return rmse_xyz, rmse_norm

def velocity_rmse(X_truth, Xhat):
    # calculates velocity rmse
    e = Xhat[:, 3:] - X_truth[:, 3:]
    rmse_v = np.sqrt(np.mean(e**2, axis=0))
    rmse_norm = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    return rmse_v, rmse_norm

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
        nees[k] = float(e.T @ np.linalg.solve(P, e))
    return nees