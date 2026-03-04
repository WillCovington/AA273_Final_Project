import numpy as np

# all of our functions that have to do with ground stations

MOON_OMEGA_RAD_S = 2.6617e-6
T0_S = 0.0

def _rotz(theta: float) -> np.ndarray:
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def R_I_to_BF(t_s: float) -> np.ndarray:
    theta = MOON_OMEGA_RAD_S * (t_s - T0_S)
    return _rotz(theta)

def define_ground_station_locations(n, lat_max_deg = 30.0, seed=None):
    # equally longitudinally spaced ground with some variance in latitude
    rng = np.random.default_rng(seed)
    locations = []
    for j in range(n):
        lon = j * (360.0/n)
        lat = rng.uniform(-lat_max_deg, lat_max_deg)
        locations.append((lat, lon))
    return locations

def gs_bodyfixed_position(gs_loc, r_moon):
    lat_deg, lon_deg = gs_loc
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    clat = np.cos(lat); slat = np.sin(lat)
    clon = np.cos(lon); slon = np.sin(lon)

    return r_moon * np.array([clat*clon, clat*slon, slat], dtype=np.float64)

def gs_state_inertial(gs_loc, t_s, model):
    r_gs_BF = gs_bodyfixed_position(gs_loc, model.r0_m)

    R_I2BF = R_I_to_BF(t_s)
    R_BF2I = R_I2BF.T

    r_gs_I = R_BF2I @ r_gs_BF

    omega_I = np.array([0.0, 0.0, MOON_OMEGA_RAD_S], dtype=np.float64)
    v_gs_I = np.cross(omega_I, r_gs_I)
    return r_gs_I, v_gs_I

def is_visible_from_station(r_sc_I, r_gs_I, elev_mask_deg=0.0):
    los = r_sc_I - r_gs_I
    u_los = los / (np.linalg.norm(los) + 1e-12)
    u_up = r_gs_I / (np.linalg.norm(r_gs_I) + 1e-12)

    # elevation = asin(u_los · u_up)
    sin_el = float(np.dot(u_los, u_up))
    sin_el = np.clip(sin_el, -1.0, 1.0)
    el_deg = np.degrees(np.arcsin(sin_el))
    return el_deg >= elev_mask_deg


