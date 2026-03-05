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