# because we'll have a variable number of ground stations, we need to be able to take in a variable number of measurements

def define_ground_station_locations(n):
    # for beginning ease, we'll have all the ground stations on the lunar equator and be equally spaced in longitude
    locations = []
    for j in range(n):
        longitude = j * (360 / n)
        latitude = 0
        locations.append((latitude, longitude))
    return locations

def take_measurements(state, ground_station_locations, noise=True):
    # for each ground station, we will take a measurement of the range and range rate to the spacecraft
    measurements = []
    