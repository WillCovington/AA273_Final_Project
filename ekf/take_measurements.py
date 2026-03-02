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

# I'm starting to second guess if we need this function like I imagined, but I'll fix it later
def take_measurements(state, ground_station_locations, R, noise=True):
    # for each ground station, we will take a measurement of the range and range rate to the spacecraft
    # TODO still need to get the coordinate transformations sorted
    measurements = []
    x = state[:3] # "true" position vector
    v = state[3:6] # "true" velocity vector
    # checking for noise
    if noise:
        w = np.multivariate_normal(mean = np.zeros(2), cov=R)
    else:
        w = np.zeros(2)
    for gs_loc in ground_station_locations:
        # converting our ground station location from lat, long, radius to xyz 
        # TODO: still need to work out if we're working in LCLF or LCI
        gs_loc_xyz = lat_long_radius_to_xyz(gs_loc) # need to write this function later
        gs_vel_xyz = gs_xyz_to_vel(gs_loc_xyz) # this should simply be a radial velocity problem if we assume constant radius and angfular velocity
        # taking the range measurement for our current state and the ith ground station
        range = x - gs_loc_xyz # range vector
        relative_velocity = v - gs_vel_xyz # 
        range_measurement = np.linalg.norm(range) + w[0] # actual range measurement (just a number)
        # using the range measurement, we can calculate the range rate
        range_rate_measurement = np.dot(range, relative_velocity) / (np.linalg.norm(range) + 1e-10) + w[1]
        measurements.append((range_measurement, range_rate_measurement))
    return np.array(measurements).T # returning as a 2 x m array, where m is the number of ground stations