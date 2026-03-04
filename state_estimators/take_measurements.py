# because we'll have a variable number of ground stations, we need to be able to take in a variable number of measurements
import numpy as np

def define_ground_station_locations(n):
    # for beginning ease, we'll have all the ground stations on the lunar equator and be equally spaced in longitude
    locations = []
    for j in range(n):
        longitude = j * (360 / n)
        latitude = 0
        locations.append((latitude, longitude))
    return locations

# I realize that I have a bad set of names here, but this is the jist
# take_measurements_ekf: this is our ideal set of measurements, and only returns range and range rate so we can use it in our EKF
# generate_measurements: this is what we run right at the beginning of our program after our state has been propogated, and it contains the full spread of measurement-related information and includes noise
def take_measurements_ekf(state, ground_station_locations, time, R, noise=True):
    # for each ground station, we will take a measurement of the range and range rate to the spacecraft
    measurements = []
    r = np.asarray(state[:3]) # "true" position vector
    v = np.asarray(state[3:6]) # "true" velocity vector

    m = len(ground_station_locations) # number of ground stations
    y = np.zeros(2*m, dtype=np.float64)

    for i, gs_loc in enumerate(ground_station_locations):
        # checking for noise
        if noise:
            if R is None:
                raise ValueError("R hasn't been provided, please include a measurement covariance matrix of size 2x2")
            w = np.multivariate_normal(mean = np.zeros(2), cov=R)
        else:
            w = np.zeros(2)

        # rotate to lunar centered inertial
        gs_r_I, gs_v_I = gs_state_inertial(gs_loc, time) # need to write this function later
        
        # taking the range measurement for our current state and the ith ground station
        range_vector = r - gs_r_I # range vector
        range_mag = np.linalg.norm(range_vector)
        relative_velocity = v - gs_v_I 
        range_measurement = range_mag + w[0] # actual range measurement (just a number)
        # using the range measurement, we can calculate the range rate
        range_rate_measurement = np.dot(range_vector, relative_velocity) / (range_mag + 1e-10) + w[1]
        # now adding to our measurement vector, y
        y[2*i] = range_measurement
        y[2*i + 1] = range_rate_measurement
    return y

def generate_measurements(t_grid, X_truth, gs_locations, model, sigma_rho = 5.0, sigma_rhodot = 0.05, elev_mask_deg = 0.0, add_noise = True, seed = None):
    # this is what we'll call at the very beginning of our ekf process
    # basically, this is what generates our set of actual ground station measurements, not the ideal ones for the ekf like up above
    # the ouput is just a list of dictionaries, measurements[k] = {"t": t_k, "y": y_k, "gs": gs_locations}
    # y_k is shape (2 * Ngs, ) where we have Nans for "invisible" stations
    rng = np.random(seed)

    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1,)
    X_truth = np.asarray(X_truth, dtype=np.float64)
    if X_truth.shape[0] != t_grid.size or X_truth.shape[1] != 6:
        raise ValueError("X_truth must be shape (len(t_grid), 6)")

    Ngs = len(gs_locations)
    R_full = build_R_full(Ngs, sigma_rho, sigma_rhodot)

    measurements = []

    for k, t_k in enumerate(t_grid):
        x_true = X_truth[k]
        r_sc = x_true[:3]
        v_sc = x_true[3:]

        # Always output fixed-size y (2*Ngs,)
        y = np.full(2*Ngs, np.nan, dtype=np.float64)

        for i, gs_loc in enumerate(gs_locations):
            r_gs, v_gs = gs_state_inertial(gs_loc, float(t_k), model)

            # visibility test
            if not is_visible_from_station(r_sc, r_gs, elev_mask_deg=elev_mask_deg):
                continue  # leave NaNs

            rho_vec = r_sc - r_gs
            rho = np.linalg.norm(rho_vec) + 1e-12
            u = rho_vec / rho
            vrel = v_sc - v_gs
            rhodot = float(np.dot(u, vrel))

            if add_noise:
                # independent noise per station (consistent with R_full diag)
                rho += rng.normal(0.0, sigma_rho)
                rhodot += rng.normal(0.0, sigma_rhodot)

            y[2*i] = rho
            y[2*i+1] = rhodot

        measurements.append({
            "t": float(t_k),
            "y": y,
            "gs": gs_locations,  # fixed list/order, reused each epoch
            "R_full": R_full     # optional convenience
        })

    return measurements


def is_visible_from_station(r_sc_I, r_gs_I, elev_mask_deg=0.0):
    # takes in a list of all our measurements and determines if the satellite is actually visible from a given ground station
    los = r_sc_I - r_gs_I
    los_norm = np.linalg.norm(los) + 1e-12
    u_los = los / los_norm

    u_up = r_gs_I / (np.linalg.norm(r_gs_I) + 1e-12)  # outward normal at station

    # elevation angle: el = asin(u_los · u_up)
    sin_el = float(np.dot(u_los, u_up))
    sin_el = np.clip(sin_el, -1.0, 1.0)
    el_deg = np.degrees(np.arcsin(sin_el))

    return el_deg >= elev_mask_deg

def build_R_full(Ngs, sigma_rho, sigma_rhodot):
    # since we'll have varying amounts of ground station coverage, we may need to resize R depending on what's visible
    R_full = np.zeros((2*Ngs, 2*Ngs), dtype=np.float64)
    for i in range(Ngs):
        R_full[2*i, 2*i] = sigma_rho**2
        R_full[2*i+1, 2*i+1] = sigma_rhodot**2
    return R_full

def filter_valid_measurements(y, y_hat, C, R_full):
    # takes in a set of all measurements and determines which ones are and aren't valid
    y = np.asarray(y).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)

    valid = np.isfinite(y)
    if not np.any(valid):
        return None

    yv = y[valid]
    y_hatv = y_hat[valid]
    Cv = C[valid, :]
    Rv = R_full[np.ix_(valid, valid)]
    return yv, y_hatv, Cv, Rv

####################################################
# everything under here is for converting to our inertial frame from our lat-lon frame
MOON_OMEGA_RAD_S = 2.6617e-6 # moon angular velocity
T0_S = 0.0 # initial time

def rotz(theta: float) -> np.ndarray:
    # returns a rotation matrix around i3
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1]])
    return rot

def R_I_to_BF(t_s: float) -> np.ndarray:
    # takes in a time and returns the corresponding rotation matrix to go from intertial to Lunar Body Fixed
    theta = MOON_OMEGA_RAD_S * (t_s - T0_S) # angle traveled
    return rotz(theta)

def gs_bodyfixed_position(gs_loc, r_moon):
    # Inputs:
    # gs_loc: either (lat_deg, lon_deg) or (lat_deg, lon_deg, alt_m) for our ground station
    # r_moon: lunar radius [m]
    # Outputs:
    # returns the lunar body frame position vector for any given ground station

    # TODO currently I think gs_loc is just latitude and longitude, but I should make sure later
    if len(gs_loc) == 2:
        lat_deg, lon_deg = gs_loc
        alt_m = 0.0
    else:
        lat_deg, lon_deg, alt_m = gs_loc
    
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    clat = np.cos(lat); slat = np.sin(lat)
    clon = np.cos(lon); slon = np.sin(lon)

    r = r_moon + alt_m 
    return r * np.array([clat * clon, clat * slon, slat], dtype=np.float64)

def gs_state_inertial(gs_loc, t_s, model):
    # takes in a ground station location in lat, lon, at a given time t_s and returns the LCI location for the ground station
    # Inputs:
    # gs_loc: probably (lat, lon) [deg] for a given gs location
    # t_s: the time at which the inertial check is being made, needed to determine the angle the moon has rotated to
    # model: needed to determine the lunar radius
    # Output:
    # r_gs_I: position vector for the ground station in the inertial frame
    # v_gs_I: velocity vector for the ground station in the inertial frame
    r_gs_BF = gs_bodyfixed_position(gs_loc, model.r0_m)
    
    R_I2BF = R_I_to_BF(t_s) # retrieves our rotation matrix from inertial to body fixed
    R_BF2I = R_I2BF.T # now we have our inverse rotation matrix

    r_gs_I = R_BF2I @ r_gs_BF # inertial position vector

    omega_I = np.array([0.0, 0.0, MOON_OMEGA_RAD_S], dtype=np.float64) # rotation rate of moon
    v_gs_I = np.cross(omega_I, r_gs_I) # inertial velocity vector

    return r_gs_I, v_gs_I

