import numpy as np
from gnss_satellites import gnss_state_inertial, is_visible_gnss, R_MOON


def build_R_full_gnss(Ntx, sigma_rho, sigma_rhodot):
    # builds our R matrix depending on how many satellites we're working with
    R_full = np.zeros((2 * Ntx, 2 * Ntx), dtype=np.float64)
    for i in range(Ntx):
        R_full[2*i, 2*i] = sigma_rho**2
        R_full[2*i+1, 2*i+1] = sigma_rhodot**2
    return R_full


def take_gnss_measurements(
    state,
    gnss_sats,
    time,
    sigma_rho=5.0,
    sigma_rhodot=0.05,
    add_noise=True,
    rng=None,
    occultation_margin_m=0.0,
):
    # works similarly to our ground station measurements, pretty much verbatim actually

    if rng is None:
        rng = np.random.default_rng()

    r_sc = np.asarray(state[:3], dtype=np.float64)
    v_sc = np.asarray(state[3:6], dtype=np.float64)

    Ntx = len(gnss_sats)
    y = np.full(2 * Ntx, np.nan, dtype=np.float64)

    for i, sat in enumerate(gnss_sats):
        r_tx, v_tx = gnss_state_inertial(sat, time)

        if not is_visible_gnss(r_sc, r_tx, r_moon=R_MOON, margin_m=occultation_margin_m):
            continue

        rho_vec = r_sc - r_tx
        rho = np.linalg.norm(rho_vec) + 1e-12
        u = rho_vec / rho
        v_rel = v_sc - v_tx
        rhodot = float(np.dot(u, v_rel))

        if add_noise:
            rho += rng.normal(0.0, sigma_rho)
            rhodot += rng.normal(0.0, sigma_rhodot)

        y[2*i] = rho
        y[2*i + 1] = rhodot

    return y


def calculate_C_gnss(gnss_sats, state_estimate, time):
    # again, works almost identically to our ground station C calculation
    Ntx = len(gnss_sats)
    C = np.zeros((2 * Ntx, 6), dtype=np.float64)

    r = np.asarray(state_estimate[0:3], dtype=np.float64)
    v = np.asarray(state_estimate[3:6], dtype=np.float64)

    for i, sat in enumerate(gnss_sats):
        r_tx, v_tx = gnss_state_inertial(sat, time)

        rho_vec = r - r_tx
        rho = np.linalg.norm(rho_vec) + 1e-12
        u = rho_vec / rho
        v_rel = v - v_tx

        # range row
        H_rho = np.zeros((1, 6))
        H_rho[0, 0:3] = u

        # range-rate row
        P = np.eye(3) - np.outer(u, u)
        dr = (P @ v_rel) / rho

        H_rhodot = np.zeros((1, 6))
        H_rhodot[0, 0:3] = dr
        H_rhodot[0, 3:6] = u

        C[2*i, :] = H_rho
        C[2*i + 1, :] = H_rhodot

    return C


def filter_valid_gnss(y, y_hat, C, R_full):
    # simply checks if our measurements are valid, again like our ground station version
    valid = np.isfinite(y)
    if not np.any(valid):
        return None

    return (
        y[valid],
        y_hat[valid],
        C[valid, :],
        R_full[np.ix_(valid, valid)]
    )