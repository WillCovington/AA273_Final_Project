import numpy as np
from dataclasses import dataclass

# the purpose of this script is to set up all of our gnss satellites so we can easily work with them in whatever reference frame we need when we start propogating

import numpy as np
from dataclasses import dataclass


# first we just introduce gravitational constants and radii and stuff

MU_EARTH = 3.986004418e14 # [m^3/s^2]
R_EARTH = 6378.1363e3 # [m]
R_MOON = 1737.4e3 # [m]

# average moon-earth distance -- doesn't factor in actual earth-moon orbit yet
EARTH_MOON_DISTANCE = 384400e3 # [m]

# placing Earth just along the x-axis to begin with
EARTH_POS_MCI = np.array([EARTH_MOON_DISTANCE, 0.0, 0.0], dtype=np.float64)


# useful rotation matrices along i1 and i3 (not respectively)

def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def rotx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]], dtype=np.float64)


# data class for all of our satellites! each one will be able to be identified by an id, a plane number, an id In the plane, their semi-mahor axis, inclination, longitude of the ascending node, and mean anomaly

@dataclass
class GNSSSatellite:
    sat_id: str
    plane_id: int
    sat_in_plane: int
    sma_m: float
    inc_deg: float
    raan_deg: float
    mean_anom0_deg: float


# now we actually set up our constellation, evenly dividing n many satellites amongst m many planes and spacing them equally to begin with

def define_gnss_constellation(
    n_planes=6,
    sats_per_plane=4,
    sma_m=26560e3,     # GPS-like semimajor axis
    inc_deg=55.0,
):
    # we'll just be returning a list of GNSSSatellite objects, each of which is defined with the signifiers above
    sats = []

    for p in range(n_planes):
        raan_deg = p * (360.0 / n_planes)

        for s in range(sats_per_plane):
            mean_anom0_deg = s * (360.0 / sats_per_plane)

            sat_id = f"P{p:02d}_S{s:02d}"

            sats.append(
                GNSSSatellite(
                    sat_id=sat_id,
                    plane_id=p,
                    sat_in_plane=s,
                    sma_m=float(sma_m),
                    inc_deg=float(inc_deg),
                    raan_deg=float(raan_deg),
                    mean_anom0_deg=float(mean_anom0_deg),
                )
            )

    return sats


# next, we have state indicators. because we're just using basic estimations between the earth and moon position to begin with, this consists of doing some simple rotationsn and translations

def gnss_state_eci(sat: GNSSSatellite, t_s: float):
    # our satellites are all in MEO, geosynchronous orbits, so this is an easy rotation
    a = sat.sma_m
    inc = np.deg2rad(sat.inc_deg)
    raan = np.deg2rad(sat.raan_deg)
    M0 = np.deg2rad(sat.mean_anom0_deg)

    # Circular orbit: mean motion = true motion
    n = np.sqrt(MU_EARTH / a**3)
    nu = M0 + n * t_s

    # Position / velocity in perifocal frame
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

    # Argument of perigee = 0 for circular simplification
    Q = rotz(raan) @ rotx(inc)

    r_eci = Q @ r_pf
    v_eci = Q @ v_pf

    return r_eci, v_eci


def gnss_state_inertial(sat: GNSSSatellite, t_s: float):
    # again, we're considering the moon and earth to be stationary relative to one another
    # later, we'll worry about orbits and relative motion but right now we're just adding vectors
    r_eci, v_eci = gnss_state_eci(sat, t_s)

    # (MCI meaning Moon Centered Inertial)
    r_mci = EARTH_POS_MCI + r_eci
    v_mci = v_eci.copy()

    return r_mci, v_mci



# one of the last things is figuring out when our satellite in LLO is actually visible to GNSS sats
def is_visible_gnss(r_sc_mci, r_tx_mci, r_moon=R_MOON, margin_m=0.0):
    
    # first, we take our satellite in LLO and our GNSS satellite, and place them both in MCI 
    r_sc = np.asarray(r_sc_mci, dtype=np.float64).reshape(3,)
    r_tx = np.asarray(r_tx_mci, dtype=np.float64).reshape(3,)

    # relative vector between the two
    d = r_tx - r_sc
    d2 = np.dot(d, d)
    if d2 < 1e-12:
        return False

    # Closest point on the segment from spacecraft to transmitter to lunar center
    # Lunar center is origin in MCI.
    tau = -np.dot(r_sc, d) / d2
    tau = np.clip(tau, 0.0, 1.0)

    closest = r_sc + tau * d
    dist_closest = np.linalg.norm(closest)

    return dist_closest > (r_moon + margin_m)