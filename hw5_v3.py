import numpy as np
from scipy.linalg import block_diag
from numpy.random import multivariate_normal as mvn
import math

# Constants
dt = 0.2  # Time step
L = 2  # Wheelbase
gx_ref = 2.3
gy_ref = 3.0

# Noise parameters
sigma_a = 0.05
sigma_fwd = 0.01
sigma_side = 0.02
sigma_theta = 0.001
sigma_v = 0.04
sigma_psi = 0.05
sigma_g = 5.0

# Initial state distribution
px_0 = 0  # initial x position
py_0 = 0  # initial y position
theta_0 = 0  # initial orientation
v_0 = 0  # initial velocity
phi_0 = 0  # initial steering angle

# Initial covariance matrix
P_0 = np.diag([10000**2, 10000**2, (2*np.pi)**2, 0, 0])

# State transition function f(x,u)
def f(x, u):
    px, py, theta, v, phi = x
    a, psi = u
    px_dot = v * np.cos(theta)
    py_dot = v * np.sin(theta)
    theta_dot = (v / L) * np.tan(phi)
    v_dot = a
    phi_dot = psi
    return np.array([px_dot, py_dot, theta_dot, v_dot, phi_dot])

# Measurement function h(x)
def h(x):
    px, py, theta, v, phi = x
    v1 = v
    v2 = (v / L) * np.tan(phi)
    v3 = px + gx_ref*np.cos(theta) - gy_ref*np.sin(theta)
    v4 = py + gy_ref*np.cos(theta) + gx_ref*np.sin(theta)
    return np.array([v1, v2, v3, v4])

# Jacobian of f(x,u) with respect to x
def F_jacobian(x, u):
    _, _, theta, v, phi = x
    return np.array([
        [0, 0, -v * np.sin(theta), np.cos(theta), 0],
        [0, 0, v * np.cos(theta), np.sin(theta), 0],
        [0, 0, 0, np.tan(phi) / L, (v * (1/np.cos(phi))**2) / L],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

# Jacobian of f(x,u) with respect to u
def G_jacobian(x, u):
    return np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])

# Jacobian of h(x) with respect to x
def H_jacobian(x):
    px, py, theta, v, phi = x
    sec_phi = 1 / np.cos(phi)
    return np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, (np.tan(phi) / L), (v * sec_phi**2 / L)],
        [dt, 0, -gx_ref*np.sin(theta) - gy_ref*np.cos(theta), 0, 0],
        [0, dt, gx_ref*np.cos(theta) - gy_ref*np.sin(theta), 0, 0]
    ])

# State noise covariance matrix Q
def Q(x, u):
    _, _, _, v, _ = x
    a, _ = u
    return np.diag([
        (sigma_fwd * abs(v))**2,
        (sigma_side * abs(v))**2,
        (sigma_theta * abs(v))**2,
        (sigma_a * abs(a))**2,
        0
    ])

# Measurement noise covariance matrix R
def R(x):
    _, _, _, v, _ = x
    return np.diag([
        (sigma_v * abs(v))**2,
        sigma_psi**2,
        sigma_g**2,
        sigma_g**2
    ])
# Extended Kalman Filter implementation
# Defining the EKF predict and update functions
def ekf_predict(x, P, u):
    # Predict next state
    f_x_u = f(x, u) * dt
    x_pred = x + f_x_u
    # Linearize the system dynamics at the current state
    F_t = F_jacobian(x, u) * dt
    # Linearize the control input
    G_t = G_jacobian(x, u) * dt
    # Predict next covariance
    Q_t = Q(x, u) * dt**2
    P_pred = F_t @ P @ F_t.T + G_t @ Q_t @ G_t.T
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z):
    # Compute the Kalman gain
    H_t = H_jacobian(x_pred)
    R_t = R(x_pred)
    S = H_t @ P_pred @ H_t.T + R_t
    K = P_pred @ H_t.T @ np.linalg.inv(S)
    # Update the state
    z_pred = h(x_pred)
    z_resid = z - z_pred
    x_upd = x_pred + K @ z_resid
    # Update the covariance
    P_upd = (np.eye(len(P_pred)) - K @ H_t) @ P_pred
    return x_upd, P_upd

import numpy as np
import matplotlib.pyplot as plt  # Optional, for plotting

# Constants given in the problem
# (Your constants here)

# Initial state covariance matrix P and other initializations
# (Your initialization code here)

# Function definitions (f, h, F_t, H_t, Sigma_xt, etc.)
# (Your function definitions here)

# Read control/observation pairs from file
def read_controls_observations(filename):
    controls = []
    observations = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            controls.append([float(parts[0]), float(parts[1])])
            observations.append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
    return controls, observations

# Read ground truth states from file
def read_ground_truth_states(filename):
    states = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            states.append([float(part) for part in parts])
    return states

controls, observations = read_controls_observations('controls_observations1.txt')
ground_truth_states = read_ground_truth_states('ground_truth_states1.txt')

# Initialize state and covariance matrix
x = np.array([px_0, py_0, theta_0, v_0, phi_0])  # Initial state
P = P_0  # Initial covariance matrix

# Placeholder for estimated states
estimated_states = [x]

# EKF Algorithm
for u, z in zip(controls, observations):
    # Prediction step
    x_pred, P_pred = ekf_predict(x, P, u)
    
    # Update step
    x, P = ekf_update(x_pred, P_pred, z, H_jacobian)
    
    # Store the estimated state
    estimated_states.append(x)

# Optional: Plotting the estimated states against the ground truth
# (Plotting code here, if desired)

print("EKF algorithm completed with the imported data.")