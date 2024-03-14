import numpy as np
import matplotlib.pyplot as plt  # Optional, for plotting


# Constants given in the problem
delta_t = 0.2  # time step
L = 2.0  # characteristic length
sigma_a = 0.05  # std dev of acceleration
sigma_fwd = 0.01  # std dev of forward velocity
sigma_side = 0.02  # std dev of sideways velocity
sigma_theta_dot = 0.001  # std dev of angular rate
sigma_v = 0.04  # std dev of velocity measurements
sigma_g = 5.0  # std dev of gps measurements

# Initial state distribution
px_0 = 0  # initial x position
py_0 = 0  # initial y position
theta_0 = 0  # initial orientation
v_0 = 0  # initial velocity
phi_0 = 0  # initial steering angle

# Initial state covariance matrix P
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
    v_meas = v
    theta_dot_meas = (v / L) * np.tan(phi)
    gx_ref = 2.3
    gy_ref = 0.3
    x_pos_meas = np.cos(theta) * px - np.sin(theta) * py + gx_ref
    y_pos_meas = np.sin(theta) * px + np.cos(theta) * py + gy_ref
    return np.array([v_meas, theta_dot_meas, x_pos_meas, y_pos_meas])

# Jacobians Ft and Ht
def F_t(x, u):
    _, _, theta, v, phi = x
    tan_phi = np.tan(phi)
    sec_phi_squared = (1 + tan_phi**2)
    return np.array([
        [0, 0, -v * np.sin(theta), np.cos(theta), 0],
        [0, 0, v * np.cos(theta), np.sin(theta), 0],
        [0, 0, 0, tan_phi / L, v * sec_phi_squared / L],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])

def H_t(x):
    px, py, theta, v, phi = x
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_phi = np.tan(phi)
    sec_phi_squared = (1 + tan_phi**2)
    gx_ref = 2.3
    gy_ref = 3.0
    return np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, tan_phi / L, v * sec_phi_squared / L],
        [delta_t, -0, -gx_ref*sin_theta - gy_ref*cos_theta, 0, 0],
        [0, delta_t, gx_ref*cos_theta - gy_ref*sin_theta, 0, 0]
        ])

# Control input Jacobian g_t
g_t = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])

# Process noise covariance matrix Sigma_xt (Σ_x,t)
def Sigma_xt(v, a):
    return np.diag([
        (sigma_fwd * np.abs(v))**2,
        (sigma_side * np.abs(v))**2,
        (sigma_theta_dot * np.abs(v))**2,
        (sigma_a * np.abs(a))**2,
        0
    ])

# Measurement noise covariance matrix Sigma_zt (Σ_z,t)
Sigma_zt = np.diag([
    (sigma_v * np.abs(v_0))**2,
    sigma_theta_dot**2,
    sigma_g**2,
    sigma_g**2
])

#prediction step

def predict(x, P, u):
    # Compute the predicted state
    x_pred = f(x, u)
    
    # Compute the Jacobian of the state transition function
    Ft = F_t(x, u)
    
    # Compute the process noise covariance for the control input
    v, a = x[3], u[0]  # Assuming u = [a, delta_phi]
    Sigma_xt_cur = Sigma_xt(v, a)
    
    # Compute the predicted covariance
    P_pred = Ft @ P @ Ft.T + Sigma_xt_cur
    
    return x_pred, P_pred

def update(x_pred, P_pred, z, H):
    # Compute the measurement prediction
    z_pred = h(x_pred)
    
    # Compute the Jacobian of the measurement function
    Ht = H_t(x_pred)
    
    # Compute the Kalman Gain
    S = Ht @ P_pred @ Ht.T + Sigma_zt  # Measurement prediction covariance
    K = P_pred @ Ht.T @ np.linalg.pinv(S)  # Kalman Gain
    
    # Update the state estimate
    x_update = x_pred + K @ (z - z_pred)
    
    # Update the covariance estimate
    I = np.eye(P_pred.shape[0])  # Identity matrix
    P_update = (I - K @ Ht) @ P_pred
    
    return x_update, P_update


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
    x_pred, P_pred = predict(x, P, u)
    
    # Update step
    x, P = update(x_pred, P_pred, z, H_t)
    
    # Store the estimated state
    estimated_states.append(x)

# Optional: Plotting the estimated states against the ground truth
# (Plotting code here, if desired)

print("EKF algorithm completed with the imported data.")

# # Plot the comparison of the mean state estimate trajectory compared to the ground truth.

# # Assuming estimated_states is a list of numpy arrays where each array
# # represents the state vector at a time step, and ground_truth_states
# # is a list of numpy arrays of the ground truth states.

# # Extract the mean state trajectory for each state component
# estimated_px = [state[0] for state in estimated_states]
# estimated_py = [state[1] for state in estimated_states]

# # Do the same for the ground truth data
# ground_truth_px = [state[0] for state in ground_truth_states]
# ground_truth_py = [state[1] for state in ground_truth_states]

# # Generate time points for the x axis
# time_points = np.arange(0, len(estimated_px) * delta_t, delta_t)

# # Create a figure and axis for the plot
# fig, ax = plt.subplots()

# # Plot the estimated trajectory
# ax.plot(time_points, estimated_px, label='Estimated px', color='blue')
# ax.plot(time_points, estimated_py, label='Estimated py', color='red')

# # Plot the ground truth trajectory
# ax.plot(time_points, ground_truth_px, label='Ground Truth px', color='blue', linestyle='dashed')
# ax.plot(time_points, ground_truth_py, label='Ground Truth py', color='red', linestyle='dashed')

# # Add labels and legend
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('State')
# ax.set_title('Comparison of EKF State Estimates and Ground Truth')
# ax.legend()

# # Show the plot
# plt.show()

# # Convert estimated_states and ground_truth_states to numpy arrays for easier manipulation
# estimated_states_array = np.array(estimated_states)
# ground_truth_states_array = np.array(ground_truth_states)

# # Calculate absolute errors for each state dimension
# absolute_errors = np.abs(estimated_states_array - ground_truth_states_array)

# # Calculate standard deviations (square root of the diagonal elements of the covariance matrix P)
# std_deviations = np.sqrt(np.diagonal(P)).reshape(-1, 1)

# # Extract absolute errors and standard deviations for each state component
# absolute_errors_px = absolute_errors[:, 0]
# absolute_errors_py = absolute_errors[:, 1]
# absolute_errors_theta = absolute_errors[:, 2]
# absolute_errors_v = absolute_errors[:, 3]
# absolute_errors_phi = absolute_errors[:, 4]

# std_dev_px = std_deviations[0]
# std_dev_py = std_deviations[1]
# std_dev_theta = std_deviations[2]
# std_dev_v = std_deviations[3]
# std_dev_phi = std_deviations[4]

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 6))

# time_points = np.arange(0, len(absolute_errors_px) * delta_t, delta_t)

# # Plot absolute errors and standard deviations for all state components on the same plot
# ax.plot(time_points, absolute_errors_px, label='Abs Error px')
# ax.plot(time_points, std_dev_px * np.ones_like(time_points), label='Std Dev px', linestyle='--')

# ax.plot(time_points, absolute_errors_py, label='Abs Error py')
# ax.plot(time_points, std_dev_py * np.ones_like(time_points), label='Std Dev py', linestyle='--')

# ax.plot(time_points, absolute_errors_theta, label='Abs Error theta')
# ax.plot(time_points, std_dev_theta * np.ones_like(time_points), label='Std Dev theta', linestyle='--')

# ax.plot(time_points, absolute_errors_v, label='Abs Error v')
# ax.plot(time_points, std_dev_v * np.ones_like(time_points), label='Std Dev v', linestyle='--')

# ax.plot(time_points, absolute_errors_phi, label='Abs Error phi')
# ax.plot(time_points, std_dev_phi * np.ones_like(time_points), label='Std Dev phi', linestyle='--')

# ax.set_ylim([0, 10])
# ax.set_title('Absolute Error and Std Dev for State Components')
# ax.set_xlabel('Time (s)')
# ax.legend()

# plt.tight_layout()
# plt.show()

