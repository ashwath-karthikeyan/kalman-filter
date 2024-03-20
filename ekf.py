from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
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

# Time step
dt = 0.2

# Dimension of the state and measurements
dim_x = 5
dim_z = 4

# Create the filter
my_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

# Initial state
my_filter.x = np.array([px_0, py_0, theta_0, v_0, phi_0]).reshape(dim_x, 1)

# State transition matrix (F)
my_filter.F = np.array([
    [1, 0, -v_0*np.sin(theta_0)*dt, np.cos(theta_0)*dt, 0],
    [0, 1, v_0*np.cos(theta_0)*dt, np.sin(theta_0)*dt, 0],
    [0, 0, 1, np.tan(phi_0)/L*dt, v_0*dt*(1+np.tan(phi_0)**2)/L],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

# Measurement function (H)
my_filter.H = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 0, np.tan(phi_0)/L, v_0*dt*(1+np.tan(phi_0)**2)/L],
    [1, 0, -2.3*np.sin(theta_0) - 0.3*np.cos(theta_0), 0, 0],
    [0, 1, 2.3*np.cos(theta_0) - 0.3*np.sin(theta_0), 0, 0]
])

# Initial covariance matrix (P)
my_filter.P = P_0

# Measurement noise (R)
my_filter.R = np.diag([sigma_v**2, sigma_theta_dot**2, sigma_g**2, sigma_g**2])

#Model incosistency matrix (Q)
Q_var = np.diag([(sigma_fwd)**2, (sigma_side)**2, (sigma_theta_dot)**2, (sigma_a)**2, 0])

my_filter.Q = Q_var

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

# EKF Algorithm
estimated_states = []  # Initialize list to store estimated states
for u, z in zip(controls, observations):
    # Prediction step
    my_filter.predict(u)
    
    # Update step
    my_filter.update(z)

    # Store the estimated state
    estimated_states.append(my_filter.x)

# Extract the state value for each state component, in this case x and y
estimated_px = [state[0] for state in estimated_states]
estimated_py = [state[1] for state in estimated_states]

# Do the same for the ground truth data
ground_truth_px = [state[0] for state in ground_truth_states]
ground_truth_py = [state[1] for state in ground_truth_states]

# Generate time points for the x axis
time_points = np.arange(0, len(estimated_px) * delta_t, delta_t)

# Adjust time_points to match the length of the ground truth data
time_points_ground_truth = np.arange(0, len(ground_truth_px) * delta_t, delta_t)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the estimated trajectory
ax.plot(time_points, estimated_px, label='Estimated px', color='blue')
ax.plot(time_points, estimated_py, label='Estimated py', color='red')

# Plot the ground truth trajectory using the adjusted time_points_ground_truth
ax.plot(time_points_ground_truth, ground_truth_px, label='Ground Truth px', color='blue', linestyle='dashed')
ax.plot(time_points_ground_truth, ground_truth_py, label='Ground Truth py', color='red', linestyle='dashed')

# Add labels and legend
ax.set_xlabel('Time (s)')
ax.set_ylabel('State')
ax.set_title('Comparison of EKF State Estimates and Ground Truth')
ax.legend()

# Show the plot
plt.show()