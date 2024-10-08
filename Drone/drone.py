import gymnasium as gym  # Update to gymnasium
from gymnasium import spaces  # Update to gymnasium
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# Define the Drone Environment
class DroneEnv(gym.Env):  # Inherit from gymnasium.Env
    """Custom Environment for Drone Stabilization and Navigation with forward movement."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(DroneEnv, self).__init__()

        # Action space: 6 actions (turn left, turn right, move forward, stabilize, up, down)
        self.action_space = spaces.Discrete(6)
        
        # Observation space: Drone's x, y coordinates, orientation, wind force, and stability status
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0]),  # x, y, orientation (0-3), wind force, stable
            high=np.array([10, 10, 3, 1, 1]),
            dtype=np.float32
        )

        # Initialize drone position, wind, and image waypoints
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.drone_pos = np.array([5, 5], dtype=np.float32)  # Drone starts in the center
        self.orientation = 0  # 0 = north, 1 = east, 2 = south, 3 = west
        self.wind_force = random.uniform(-1, 1)  # Wind force on the drone
        self.stable = 1  # Drone is stable initially
        self.image_waypoints = [(2, 2), (8, 8)]  # Define waypoints for image capture
        self.captured_images = 0  # Images captured so far

        # Log the drone's initial position and trajectory
        self.trajectory = [self.drone_pos.copy()]

        # Return the initial state
        return np.array([self.drone_pos[0], self.drone_pos[1], self.orientation, self.wind_force, self.stable])

    def step(self, action):
        """Take an action and return the new state, reward, done, and info."""
        done = False
        reward = 0
        
        # Actions: 0 = turn left, 1 = turn right, 2 = move forward, 3 = stabilize, 4 = move up, 5 = move down
        if action == 0:
            self.orientation = (self.orientation - 1) % 4  # Turn left (counterclockwise)
        elif action == 1:
            self.orientation = (self.orientation + 1) % 4  # Turn right (clockwise)
        elif action == 2:
            self.move_forward()  # Move forward based on orientation
        elif action == 3:
            # Stabilize the drone
            self.stable = 1
            reward += 1  # Reward for stabilizing
        elif action == 4:
            self.drone_pos[1] += 1  # Move up
        elif action == 5:
            self.drone_pos[1] -= 1  # Move down

        # Add wind disturbance to the drone
        self.drone_pos += self.wind_force * np.array([1, 1])

        # Check for out-of-bounds (the drone should stay within the grid)
        self.drone_pos = np.clip(self.drone_pos, 0, 10)

        # Reduce stability if not stabilizing
        if action != 3:
            self.stable = 0
        
        # Image capture reward
        if tuple(np.round(self.drone_pos).astype(int)) in self.image_waypoints and self.stable == 1:
            self.captured_images += 1
            reward += 10  # Large reward for capturing an image at the waypoint

        # Check if all waypoints are captured
        if self.captured_images == len(self.image_waypoints):
            done = True  # Episode ends when all images are captured
            reward += 50  # Bonus for completing the task

        # Wind force randomly changes over time
        self.wind_force = random.uniform(-1, 1)

        # Log the trajectory for visualization
        self.trajectory.append(self.drone_pos.copy())

        # Observation
        obs = np.array([self.drone_pos[0], self.drone_pos[1], self.orientation, self.wind_force, self.stable])

        return obs, reward, done, {}

    def move_forward(self):
        """Move the drone forward based on its orientation."""
        if self.orientation == 0:  # North
            self.drone_pos[1] += 1
        elif self.orientation == 1:  # East
            self.drone_pos[0] += 1
        elif self.orientation == 2:  # South
            self.drone_pos[1] -= 1
        elif self.orientation == 3:  # West
            self.drone_pos[0] -= 1

    def render(self, mode='human'):
        """Render the environment (optional)."""
        print(f"Drone Position: {self.drone_pos}, Orientation: {self.orientation}, Wind Force: {self.wind_force}, Stable: {self.stable}")

    def close(self):
        pass


# Initialize the custom environment
env = DroneEnv()

# Check if the environment adheres to the Gymnasium standard
check_env(env)

# Create the PPO model (using Proximal Policy Optimization)
model = PPO('MlpPolicy', env, verbose=1)

# Setup a logger to capture training metrics
new_logger = configure("logs/", ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model (total_timesteps can be increased for better performance)
model.learn(total_timesteps=50000)

# Save the model for future use
model.save("ppo_drone")

# Load the saved model (optional, if you want to load the trained model later)
# model = PPO.load("ppo_drone")

# Visualization of the training results
def plot_results(rewards, trajectory):
    """Function to plot the results of training and drone trajectory."""
    
    # Plot rewards over time
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Rewards")
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the drone trajectory on the grid
    plt.figure(figsize=(6, 6))
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.scatter([2, 8], [2, 8], color='red', label='Waypoints')
    plt.title("Drone Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the trained model and gather data for visualization
obs = env.reset()
rewards = []
total_reward = 0
for i in range(100):
    # The model predicts the best action given the current observation
    action, _states = model.predict(obs)
    
    # Take the action and receive the new state and reward
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render the environment (prints the drone status)
    env.render()
    
    # If the episode ends (all images captured or done condition is met), reset the environment
    if done:
        rewards.append(total_reward)
        total_reward = 0
        obs = env.reset()

# Plot the results
plot_results(rewards, env.trajectory)
