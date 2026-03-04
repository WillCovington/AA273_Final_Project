# running through some function ideas
import numpy as np
from state_estimators.take_measurements import *
from gravity.dynamics import propogate


def ekf(x0, P0, measurements, model, L_max, eps, Q, R):
    # Inputs:
    # x0: our initial state (r, v) estimate (6x1)
    # P0: our initial covariance (6x6)
    # measurements: our set of measurements, each of which should have a timestamp t, measurement y (2,1), and a set of 
    x, P = x0.copy().reshape(6,), P0.copy().reshape(6,6)
    t_prev = float(measurements[0]["t"]) # this is our zero-time -- probably just gonna be zero
    gs_all= measurements[0]["gs"] # we can just store our ground station list here

    xs = [x.copy()]
    Ps = [P.copy()]
    ts = [t_prev]
    for meas in measurements[1:]:
        t_k = float(meas["t"]) # time of measurement

        y_k = np.asarray(meas["y"], dtype=float).reshape(-1,) # actual measurement, range and range rate 
                
        x, P = ekf_step(x, P, y_k, t_k, t_prev, model, gs_all, L_max, eps, Q, R)
        t_prev = t_k 

        xs.append(x.copy()); Ps.append(P.copy()); ts.append(t_k)

    return np.array(ts), np.array(xs), Ps

def ekf_step(x, P, y, time_curr, time_prev, model, gs_loc, L_max, eps, Q, R):
    # a single step of our EKF
    # inputs:
    # x: state vector [x_t|t-1, y_t|t-1, z_t|t-1, vx_t|t-1, vy_t|t-1, vz_t|t-1]
    # P: covariance matrix (6x6)
    # y: measurement vector [x_m, y_m, z_m]
    # time_curr: current time we're estimating at
    # time_prev: previous time
    # model: our propogation model
    # gs_loc: list of ground station locations
    # L_max: truncation degree
    # eps: finite difference step size

    # ouputs:
    # x_updated: our updated state estimate
    # P_updated: our updated covariance estimate

    # because we care about the time at which we receive measurements, we calculate dt as a function of the current and the previous timestep
    dt = time_curr - time_prev
    
    # Predict step
    x_pred = propogate(x, time_prev, dt,L_max, model, method="rk4", substeps=1 )
    A = calculate_A_finite_diff(dt, time_curr, x_pred[:3], L_max, model, eps)
    P_pred = A @ P @ A.T + Q

    # Update step
    C = calculate_C_gs(gs_loc, x_pred, time_curr, model) # calculates C given our predicted state
    
    # predicted measurements for all of our stations
    y_hat = take_measurements_ekf(x_pred, gs_loc, time_curr, R=0, noise=False) # our ideal measurement from our estimate, returns range and range rate

    # need to filter out instances where the satellite isn't visible for some ground stations
    filtered = filter_valid_measurements(y=y, y_hat=y_hat, C_full = C, R_full = R)
    if filtered is None: # this basically means that none of the ground stations could see the satellite
        # if that's the case, we don't do an update step
        return x_pred, P_pred
    
    yv, yhatv, Cv, Rv = filtered

    innov = yv - yhatv  # innovation (measurement - prediction)
    S = Cv @ P_pred @ Cv.T + Rv  # innovation matrix
    K = P_pred @ Cv.T @ np.linalg.inv(S)  # Kalman gain

    x_updated = x_pred + K @ innov
    P_updated = (np.eye(6) - K @ Cv) @ P_pred

    return x_updated, P_updated

def calculate_A_finite_diff(dt, t, r, L_max, model, eps=5):
    # uses a finite differencing technique to calculate our dynamics Jacobian
    # Inputs:
    # dt: the differential time step forwards we're taking for our state estimate
    # r: our current estimated position vector, given as r = [x_t|t, y_t|t, z_t|t]
    # L_max: our truncation degree, basically how precise we want to be with our spherical harmonics
    # eps: the distance we want to linearize over when creating A_bl
    # Outputs:
    # A: our dynamics jacobian (6x6)
    A_bl = accel_jacobian(r, t, L_max, model, eps)
    A = np.block([[np.eye(3), np.eye(3) * dt], [A_bl * dt, np.eye(3)]])
    
    return A

def accel_jacobian(r, t, L_max, model, eps = 0.5):
    # attempts to find the acceleration portion of our jacobian (the 3x3 bottom left block)
    # Inputs:
    # r: current estimated position vector in LCI, given as r = [x_t|t, y_t|t, z_t|t]
    # L_max: our truncation degree
    # eps: the extra oomph we wanna push our state estimate in each direction of to linearize over
    # Outputs: 
    # A: our linearized dynamics Jacobian

    r = np.asarray(r, dtype=np.float64).reshape(3,) # recasting our position vector for safety
    A = np.zeros((3, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps

        # calling on our acceleration model
        ap = model.accel_inertial(r + dr, t, L_max)
        am = model.accel_inertial(r - dr, t, L_max)

        A[:, i] = (ap - am) / (2.0 * eps) # finite differencing of the estimated acceleration at current state estimate plus epsilon minus current state estimate minus epsilon
    return A

def calculate_C_gs(gs_locations, state_estimate, time, model):
    # calculates the measurement Jacobian C using a series of ground stations on the moon
    # unlike A, C can actually be computed pretty easily analytically regardless of the sh d/o, so no need for a finite differencing technique
    # C will be a 2*m x n matrix; 2m because for each ground station, we receive both a range measurement and a range rate measurement
    m = len(gs_locations)
    # initializing C
    C = np.zeros((2*m, 6), dtype=np.float64)
    # getting our estimated state
    r = np.asarray(state_estimate[0:3], dtype=np.float64)  
    v = np.asarray(state_estimate[3:6], dtype=np.float64)

    # looping through and building C block by block
    for i, gs_loc in enumerate(gs_locations):
        # to calculate C, we need to know what specific ground station we received our range from
        gs_loc = np.asarray(gs_loc, dtype=float)
        # rotating to the inertial frame
        gs_r_xyz, gs_v_xyz = gs_state_inertial(gs_loc, time, model)
        
        # now we're going to do what take_measurements was supposed to do, but better
        range_vector = r - gs_r_xyz
        range_mag = np.linalg.norm(range_vector) + 1e-12
        u = range_vector / range_mag # we use this term a lot so I'm just defining it outright
        dv = v - gs_v_xyz

        # putting together the different parts of our C block
        C_tl = u.reshape(1, 3)
        C_tr = np.zeros((1, 3))

        C_bl = (dv / range_mag - np.dot(range_vector, dv) * range_vector / (range_mag**3)).reshape(1, 3)
        C_br = u.reshape(1, 3)
        
        C_instance = np.block([[C_tl, C_tr], [C_bl, C_br]])
        C[2*i:2*i+2, :] = C_instance
    return C
        
