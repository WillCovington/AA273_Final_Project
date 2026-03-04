import numpy as np
from gravity.gravity_model import GravityModel
from gravity.dynamics import propogate

# this is the ekf process used for estimating our satellite's state using ground stations on the moon
# once this is working, the plan is to move onto the UKF, then later onto using GNSS satellites

def main():
    # setting a random seed for consistency
    np.random.seed(134)

    # first we load our "truth" gravity model
    model = GravityModel.from_npz("clone_averages/grgm1200a_clone_mean_L660.npz")

    # set our initial conditions
    alt = 50 # altitude in km
    r_mag = model.r0_m + alt*10**3 # radius in m
    mu = model.gm_m3_s2
    v_circ = np.sqrt(mu/r_mag) # easy circular velocity conversion

    x0 = np.array([r_mag, 0.0, 0.0, 0.0, v_circ, 0.0], dtype=np.float64)
    
    # starting state covariance (may need to tune)
    P0 = np.diag([1e6, 1e6, 1e6, 1e0, 1e0, 1e0])

    # Truncation evaluation degree
    L_max = 200

    # setting up our stochasticity styff
    Q = np.eye(6) * 1e-6 # process noise covariance
    R = np.eye(2) * 1.0 # measurement noise covariance

    # TODO proopgate the system and take measurements
    # blah blah, system trajectory = ...
    # measurements = generate_measurements(system_trajectory)
    # measurements will have a set of time steps, set of actual range/range rate values for each gs, and then a list of the ground stations
    # those measurements get passed into the ekf function


if __name__ == "__main__":
    main()