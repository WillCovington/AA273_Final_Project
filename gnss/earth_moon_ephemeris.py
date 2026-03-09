import numpy as np
# this is the first script I'm adding in that tries to model the orbit of the Earth and the Moon rather than having them be stationary relative to one another
# rather than try and propogate the entirety of an actual Earth Moon orbit (which is roughly circular with an eccentricity of ~0.05), we're just going to Assume it's actually circular

# Mean Earth-Moon distance
EARTH_MOON_DISTANCE_M = 384400e3

# Sidereal month
EARTH_MOON_PERIOD_S = 27.321661 * 86400.0

# Mean angular rate
N_EM = 2.0 * np.pi / EARTH_MOON_PERIOD_S


def earth_state_mci(t_s: float, phase0_rad: float = 0.0):
    # returns the earth's state in our Moon Centered Inertial Frame, which is the basis of everything we're working in

    th = N_EM * t_s + phase0_rad

    r_earth = EARTH_MOON_DISTANCE_M * np.array([
        np.cos(th),
        np.sin(th),
        0.0
    ], dtype=np.float64)

    v_earth = EARTH_MOON_DISTANCE_M * N_EM * np.array([
        -np.sin(th),
         np.cos(th),
         0.0
    ], dtype=np.float64)

    return r_earth, v_earth