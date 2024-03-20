import numpy as np
import matplotlib.pyplot as plt

from filterpy.monte_carlo import systematic_resample

delta_t = 0.2  # time step
# Initial state distribution
px_0 = 0  # initial x position
py_0 = 0  # initial y position
theta_0 = 0  # initial orientation
v_0 = 0  # initial velocity
phi_0 = 0  # initial steering angle

# Initial state covariance matrix P
P_0 = np.diag([10000**2, 10000**2, (2*np.pi)**2, 0, 0])

class ParticleFilter:
    def __init__(self, num_particles, state_dim, observation_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self):
        # Initialize your particles here based on your system
        # For example, particles could be initialized uniformly within a range or based on a prior distribution
        return np.random.rand(self.num_particles, self.state_dim)

    def predict(self):
        # Predict the next state of the particles
        # This function should implement your system model
        # For example, a simple random walk model could be used
        self.particles += np.random.randn(self.num_particles, self.state_dim) * 0.1

    def update(self, observation):
        # Update weights based on observation
        # This function should implement your observation model
        # For example, you could calculate the distance between predicted and actual observations and convert to weights
        weights = np.random.rand(self.num_particles) # Placeholder: Replace with your actual weight update logic
        self.weights = weights / np.sum(weights)

    def resample(self):
        indexes = systematic_resample(self.weights)
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        # Estimate the current state based on particles and weights
        # For example, you could use the mean or median of the particles weighted by their weights
        return np.average(self.particles, weights=self.weights, axis=0)

    def step(self, observation):
        self.predict()
        self.update(observation)
        self.resample()
        return self.estimate()

# Example usage
num_particles = 100
state_dim = 5
observation_dim = 4

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

estimated_states = []  # Initialize list to store estimated states

#Particle Filter goes here
pf = ParticleFilter(num_particles, state_dim, observation_dim)


# Extract the state value for each state component, in this case x and y
estimated_px = [state[0] for state in estimated_states]
estimated_py = [state[1] for state in estimated_states]

# Do the same for the ground truth data
ground_truth_px = [state[0] for state in ground_truth_states]
ground_truth_py = [state[1] for state in ground_truth_states]

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Generate time points for the x axis
time_points = np.arange(0, len(estimated_px) * delta_t, delta_t)

# Adjust time_points to match the length of the ground truth data
time_points_ground_truth = np.arange(0, len(ground_truth_px) * delta_t, delta_t)

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