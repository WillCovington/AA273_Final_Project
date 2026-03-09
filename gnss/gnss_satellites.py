import numpy as np
from dataclasses import dataclass
from gnss.earth_moon_ephemeris import earth_state_mci

# this script actually does all the modeling for the gnss satellites
# had to do a bit of a reupdate from the last version on git because I wanted to try and model the Earth-Moon system more dynamically
# that being said, the orbit is still idealized as circular

# constants
MU_EARTH = 3.986004418e14  # [m^3/s^2]
R_MOON = 1737.4e3          # [m]

# rotation about the z axis
def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

# rotation about the x axis
def rotx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ], dtype=np.float64)


# helper data class to store all of our satellite info, for Each satellite
@dataclass
class GNSSSatellite:
    sat_id: str
    plane_id: int
    sat_in_plane: int
    sma_m: float
    inc_deg: float
    raan_deg: float
    mean_anom0_deg: float


# same gnss satellite constellation definition, where you simply indicate the axis, inclination spacing, number of orbital planes and satellites per plane
def define_gnss_constellation(n_planes=6, sats_per_plane=4, sma_m=26560e3, inc_deg=55.0):
    sats = []

    for p in range(n_planes):
        raan_deg = p * (360.0 / n_planes)

        for s in range(sats_per_plane):
            mean_anom0_deg = s * (360.0 / sats_per_plane)

            sats.append(
                GNSSSatellite(
                    sat_id=f"P{p:02d}_S{s:02d}",
                    plane_id=p,
                    sat_in_plane=s,
                    sma_m=float(sma_m),
                    inc_deg=float(inc_deg),
                    raan_deg=float(raan_deg),
                    mean_anom0_deg=float(mean_anom0_deg),
                )
            )

    return sats


def gnss_state_eci(sat: GNSSSatellite, t_s: float):
    # returns the gnss satellite state in the ECI frame based on how much time has passed
    a = sat.sma_m
    inc = np.deg2rad(sat.inc_deg)
    raan = np.deg2rad(sat.raan_deg)
    M0 = np.deg2rad(sat.mean_anom0_deg)

    n = np.sqrt(MU_EARTH / a**3)
    nu = M0 + n * t_s

    r_pf = np.array([
        a * np.cos(nu),
        a * np.sin(nu),
        0.0
    ], dtype=np.float64)

    v_pf = np.array([
        -a * n * np.sin(nu),
         a * n * np.cos(nu),
         0.0
    ], dtype=np.float64)

    Q = rotz(raan) @ rotx(inc)

    r_eci = Q @ r_pf
    v_eci = Q @ v_pf

    return r_eci, v_eci


def gnss_state_inertial(sat: GNSSSatellite, t_s: float):
    # returns the gnss satellite state in our more universal Moon Centered Inertial frame
    # since we're letting the earth-moon system rotate, there's a bit more vector addition that needs to be taken care of
    r_earth, v_earth = earth_state_mci(t_s)
    r_eci, v_eci = gnss_state_eci(sat, t_s)

    r_mci = r_earth + r_eci
    v_mci = v_earth + v_eci

    return r_mci, v_mci


def is_visible_gnss(r_sc_mci, r_tx_mci, r_moon=R_MOON, margin_m=0.0):
    # the same visibility check as before
    r_sc = np.asarray(r_sc_mci, dtype=np.float64).reshape(3,)
    r_tx = np.asarray(r_tx_mci, dtype=np.float64).reshape(3,)

    d = r_tx - r_sc
    d2 = np.dot(d, d)
    if d2 < 1e-12:
        return False

    tau = -np.dot(r_sc, d) / d2
    tau = np.clip(tau, 0.0, 1.0)

    closest = r_sc + tau * d
    dist_closest = np.linalg.norm(closest)

    return dist_closest > (r_moon + margin_m)