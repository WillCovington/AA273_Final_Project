# running through some function ideas
import numpy as np

def ekf_step(x, P, y, dt, gs_loc, L_max, eps, Q, R):
    # an actual single step of our EKF
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

    # Predict step
    A = calculate_A_finite_diff(dt, x[:3], L_max, eps)
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # Update step
    C = calculate_C_gs()

    innov = y - x_pred[:3]  # innovation (measurement - prediction)
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

def accel_jacobian(r, dt, L_max, eps = 0.5):
    # attempts to find the acceleration portion of our jacobian (the 3x3 bottom left block)
    # Inputs:
    # r: current estimated position vector, given as r = [x_t|t, y_t|t, z_t|t]
    
    A = np.zeros((3, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        A[:, i] = (accel_fun(r + dr, L_max) - accel_fun(r - dr, L_max)) / (2 * eps) # finite differencing of the estimated acceleration at current state estimate plus epsilon minus current state estimate minus epsilon
    return A

def calculate_A_analytical(dt):
    # the (maybe) trickier and more computationally expensive way of finding A
    # the primary structure of A is really simple -- identity alone the main diagonal, with the top right 3x3 block just being eye(3) * dt
    A = np.eye(6)
    A[0:3, 3:6] = np.eye(3) * dt

    # general steps
    # pass in truncation degree
    # compute the potential map (call upon generate_potential)
    # convert the potential from latitude, longitude, radius to xyz
    # I think this part might just be a bitch regardless
    pass

def calculate_C_gs():
    # calculates the measurement Jacobian C using a series of ground stations on the moon


