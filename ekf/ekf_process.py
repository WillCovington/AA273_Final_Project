# running through some function ideas
import numpy as np

def calculate_A_finite_diff(dt):
    # still need to find out what this process is
    A = np.eye(6)
    A[0:3, 3:6] = np.eye(3) * dt
    # this is where the hard part of finding A is
    pass

def calculate_A_analytical(dt):
    # the (maybe) trickier way of finding A
    A = np.eye(6)
    A[0:3, 3:6] = np.eye(3) * dt

    # general steps
    # pass in truncation degree
    # compute the potential map (call upon generate_potential)
    # convert the potential from latitude, longitude, radius to xyz
    # I think this part might just be a bitch regardless
    pass





