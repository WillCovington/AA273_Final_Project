import numpy as np

from gravity.dynamics import propagate
from ekf_fun import take_measurements, filter_valid


# this one symmetrizes our array P because otherwise things get tricky
def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def _safe_cholesky(P: np.ndarray, jitter0: float = 1e-12, max_tries: int = 8) -> np.ndarray:
    # this just guarantees that we don't fuck up our matrix square root if it happens to be near-singular
    P = _symmetrize(P)
    jitter = jitter0
    I = np.eye(P.shape[0])

    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(P + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError("Cholesky failed even after jittering covariance.")


def _ukf_weights(n: int, alpha: float, beta: float, kappa: float):
    # generates our UKF weights
    lam = alpha**2 * (n + kappa) - n
    c = n + lam

    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))

    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)

    return lam, c, Wm, Wc


def generate_sigma_points(x: np.ndarray, P: np.ndarray, alpha=1e-3, beta=2.0, kappa=0.0):
    # generates our sigma points (duh)
    x = np.asarray(x, dtype=np.float64).reshape(-1,)
    P = np.asarray(P, dtype=np.float64)

    n = x.size
    lam, c, Wm, Wc = _ukf_weights(n, alpha, beta, kappa)

    S = _safe_cholesky(c * P)

    Xsig = np.zeros((2 * n + 1, n), dtype=np.float64)
    Xsig[0] = x
    for i in range(n):
        col = S[:, i]
        Xsig[i + 1] = x + col
        Xsig[n + i + 1] = x - col

    return Xsig, Wm, Wc


def ukf_predict(
    x: np.ndarray,
    P: np.ndarray,
    t_prev: float,
    t_k: float,
    model,
    L_max: int,
    Q: np.ndarray,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
):
    """
    UKF prediction step.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1,)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    n = x.size
    dt = float(t_k - t_prev)

    Xsig, Wm, Wc = generate_sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)

    # Propagate each sigma point through nonlinear dynamics
    Xsig_pred = np.zeros_like(Xsig)
    for i in range(Xsig.shape[0]):
        Xsig_pred[i] = propagate(
            Xsig[i],
            t_prev,
            dt,
            L_max,
            model,
            method="rk4",
            substeps=1,
        )

    # Predicted mean
    x_pred = np.sum(Wm[:, None] * Xsig_pred, axis=0)

    # Predicted covariance
    P_pred = np.zeros((n, n), dtype=np.float64)
    for i in range(Xsig_pred.shape[0]):
        dx = (Xsig_pred[i] - x_pred).reshape(n, 1)
        P_pred += Wc[i] * (dx @ dx.T)

    P_pred += Q
    P_pred = _symmetrize(P_pred)

    return x_pred, P_pred, Xsig_pred, Wm, Wc


def ukf_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    Xsig_pred: np.ndarray,
    Wm: np.ndarray,
    Wc: np.ndarray,
    y_k: np.ndarray,
    t_k: float,
    model,
    gs_all,
    R_full: np.ndarray,
    sigma_rho_for_pred: float = 0.0,
    sigma_rhodot_for_pred: float = 0.0,
):
    """
    UKF measurement update.

    Uses the same measurement convention as your EKF:
      y = [rho1, rhodot1, rho2, rhodot2, ...]
    with NaNs for invisible stations in the actual measurement vector.
    """
    x_pred = np.asarray(x_pred, dtype=np.float64).reshape(-1,)
    P_pred = np.asarray(P_pred, dtype=np.float64)
    y_k = np.asarray(y_k, dtype=np.float64).reshape(-1,)
    R_full = np.asarray(R_full, dtype=np.float64)

    n = x_pred.size
    nsig = Xsig_pred.shape[0]

    # First get nominal predicted measurement and valid mask from actual y
    y_hat_full = take_measurements(
        x_pred,
        gs_all,
        t_k,
        model,
        sigma_rho=sigma_rho_for_pred,
        sigma_rhodot=sigma_rhodot_for_pred,
        elev_mask_deg=0.0,
        add_noise=False,
    )

    valid = np.isfinite(y_k)
    if not np.any(valid):
        return x_pred, P_pred, y_hat_full, None

    yv = y_k[valid]
    yhatv_nom = y_hat_full[valid]
    Rv = R_full[np.ix_(valid, valid)]

    m = yv.size

    # Push propagated sigma points through measurement model
    Ysig = np.zeros((nsig, m), dtype=np.float64)
    for i in range(nsig):
        yi_full = take_measurements(
            Xsig_pred[i],
            gs_all,
            t_k,
            model,
            sigma_rho=sigma_rho_for_pred,
            sigma_rhodot=sigma_rhodot_for_pred,
            elev_mask_deg=0.0,
            add_noise=False,
        )
        Ysig[i] = yi_full[valid]

    # Predicted measurement mean
    y_pred = np.sum(Wm[:, None] * Ysig, axis=0)

    # Innovation covariance and cross-covariance
    S = np.zeros((m, m), dtype=np.float64)
    Pxy = np.zeros((n, m), dtype=np.float64)

    for i in range(nsig):
        dy = (Ysig[i] - y_pred).reshape(m, 1)
        dx = (Xsig_pred[i] - x_pred).reshape(n, 1)
        S += Wc[i] * (dy @ dy.T)
        Pxy += Wc[i] * (dx @ dy.T)

    S += Rv
    S = _symmetrize(S)

    # Kalman gain
    K = Pxy @ np.linalg.inv(S)

    innov = yv - y_pred
    x_upd = x_pred + K @ innov
    P_upd = P_pred - K @ S @ K.T
    P_upd = _symmetrize(P_upd)

    update_info = {
        "valid_mask": valid,
        "innovation": innov,
        "S": S,
        "y_pred": y_pred,
        "n_valid_meas": int(m),
    }

    return x_upd, P_upd, y_hat_full, update_info


def ukf_run(
    x0,
    P0,
    measurements,
    model,
    L_max,
    Q,
    R_full,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
):
    """
    Full UKF loop, analogous to your EKF.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(6,)
    P = np.asarray(P0, dtype=np.float64).reshape(6, 6)

    ts = [float(measurements[0]["t"])]
    xs = [x.copy()]
    Ps = [P.copy()]

    t_prev = float(measurements[0]["t"])
    gs_all = measurements[0]["gs"]

    for meas in measurements[1:]:
        t_k = float(meas["t"])
        y_k = np.asarray(meas["y"], dtype=np.float64).reshape(-1,)

        # Predict
        x_pred, P_pred, Xsig_pred, Wm, Wc = ukf_predict(
            x, P, t_prev, t_k,
            model=model,
            L_max=L_max,
            Q=Q,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

        # Update
        x, P, _, _ = ukf_update(
            x_pred, P_pred, Xsig_pred, Wm, Wc,
            y_k=y_k,
            t_k=t_k,
            model=model,
            gs_all=gs_all,
            R_full=R_full,
        )

        ts.append(t_k)
        xs.append(x.copy())
        Ps.append(P.copy())
        t_prev = t_k

    return np.asarray(ts), np.vstack(xs), np.stack(Ps, axis=0)