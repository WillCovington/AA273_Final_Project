# running through some function ideas
import numpy as np
from take_measurements import *

def ekf(x0, P0, measurements, L_max, eps, Q, R):
    x, P = x0.copy(), P0.copy()
    t_prev = measurements[0]["t"] # this is our zero-time -- probably just gonna be zero

    xs = [x.copy()]
    Ps = [P.copy()]
    ts = [t_prev]
    for meas in measurements:
        t_k = meas["t"] # time of measurement
        y_k = meas["y"] # actual measurement, range and range rate
        gs_k = meas["gs"] # list of stations used for this measurement
        
        x, P = ekf_step(x, P, t_prev, t_k, y_k, gs_k, L_max, eps, Q, R)
        t_prev = t_k 

        xs.append(x.copy()); Ps.append(P.copy()); ts.append(t_k)

    return np.array(ts), np.array(xs), Ps

def ekf_step(x, P, y, time_curr, time_prev, gs_loc, L_max, eps, Q, R):
    # a single step of our EKF
    # inputs:
    # x: state vector [x_t|t-1, y_t|t-1, z_t|t-1, vx_t|t-1, vy_t|t-1, vz_t|t-1]
    # P: covariance matrix (6x6)
    # z: measurement vector [x_m, y_m, z_m]
    # dt: time step
    # L_max: truncation degree
    # eps: finite difference step size

    # ouputs:
    # x_updated: our updated state estimate
    # P_updated: our updated covariance estimate

    # because we care about the time at which we receive measurements, we calculate dt as a function of the current and the previous timestep
    dt = time_curr - time_prev
    
    # Predict step
    A = calculate_A_finite_diff(dt, x[:3], L_max, eps)
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # Update step
    C = calculate_C_gs(gs_loc, x_pred) # calculates C given our predicted state
    y_hat = take_measurements(x_pred, gs_loc, time_curr, R=0, noise=False) # our ideal measurement from our estimate, returns range and range rate

    innov = y - y_hat  # innovation (measurement - prediction)
    S = C @ P_pred @ C.T + R  # innovation matrix
    K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain

    x_updated = x_pred + K @ innov
    P_updated = (np.eye(6) - K @ C) @ P_pred

    return x_updated, P_updated

def calculate_A_finite_diff(dt, r, L_max, eps):
    # uses a finite differencing technique to calculate our dynamics Jacobian
    # Inputs:
    # dt: the differential time step forwards we're taking for our state estimate
    # r: our current estimated position vector, given as r = [x_t|t, y_t|t, z_t|t]
    # L_max: our truncation degree, basically how precise we want to be with our spherical harmonics
    # eps: the distance we want to linearize over when creating A_bl
    # Outputs:
    # A: our dynamics jacobian (6x6)
    A_bl = accel_jacobian(r, dt, L_max, eps)
    A = np.block([[np.eye(3), np.eye(3) * dt], [A_bl * dt, np.eye(3)]])
    
    return A

def accel_jacobian(r, L_max, eps = 0.5):
    # attempts to find the acceleration portion of our jacobian (the 3x3 bottom left block)
    # Inputs:
    # r: current estimated position vector, given as r = [x_t|t, y_t|t, z_t|t]
    # L_max: our truncation degree
    # eps: the extra oomph we wanna push our state estimate in each direction of to linearize over
    # Outputs: 
    # A: our linearized dynamics Jacobian

    A = np.zeros((3, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        A[:, i] = (accel_fun(r + dr, L_max) - accel_fun(r - dr, L_max)) / (2 * eps) # finite differencing of the estimated acceleration at current state estimate plus epsilon minus current state estimate minus epsilon
    return A

def calculate_C_gs(gs_locations, state_estimate):
    # calculates the measurement Jacobian C using a series of ground stations on the moon
    # unlike A, C can actually be computed pretty easily analytically regardless of the sh d/o, so no need for a finite differencing technique
    # C will be a 2*m x n matrix; 2m because for each ground station, we receive both a range measurement and a range rate measurement
    m = len(gs_locations)
    # initializing C
    C = np.zeros((2*m, 6))
    # getting our estimated state
    r = np.asarray(state_estimate[0:3], dtype=float)  
    v = np.asarray(state_estimate[3:6], dtype=float)

    # looping through and building C block by block
    for i, gs_loc in enumerate(gs_locations):
        # to calculate C, we need to know what specific ground station we received our range from
        gs_loc = np.asarray(gs_loc, dtype=float)
        # TODO still need to figure out our reference frame
        gs_r_xyz, gs_v_xyz = lat_long_radius_to_xyz(gs_loc)
        
        # now we're going to do what take_measurements was supposed to do, but better
        range_vector = r - gs_r_xyz
        range_mag = np.linalg.norm(range_vector)
        dv = v - gs_v_xyz

        # putting together the different parts of our C block
        C_tl = (range_vector / range_mag).reshape(1, 3)
        C_tr = np.zeros((1, 3))
        C_bl = (dv / range_mag - np.dot(range_vector, dv) * range_vector / (range_mag**3)).reshape(1, 3)
        C_br = (range_vector / range_mag).reshape(1, 3)
        C_instance = np.block([[C_tl, C_tr], [C_bl, C_br]])
        C[2*i:2*i+2, :] = C_instance
    return C


        
