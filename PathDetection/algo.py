import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Configuration
BASE_DIR = "SkyScenes"
HP = "H_15_P_90"
WEATHER = "ClearNoon"
TOWN = "Town07"
START_X, START_Y = 0, 0  # UAV start position
END_X, END_Y = 10, 10  # UAV end position (diagonal points)
MAX_TIME_STEPS = 100  # Max steps before termination
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # Input: x, y (current position)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 3)  # Output: 3 actions (left, forward, right)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Helper functions
def get_image_path(base_dir, hp, weather, town, x, y):
    path = os.path.join(base_dir, hp, weather, town, f"{x}_{y}.png")
    if os.path.exists(path):
        return path  # Image exists
    else:
        return None  # Out of bounds

def is_out_of_bounds(x, y):
    return get_image_path(BASE_DIR, HP, WEATHER, TOWN, x, y) is None

def calculate_reward(x, y, end_x, end_y, time_steps):
    distance_to_goal = abs(end_x - x) + abs(end_y - y)
    if distance_to_goal == 0:
        return 100  # Positive reward for reaching goal
    else:
        reward = -distance_to_goal  # Negative reward based on distance
        reward -= time_steps * 0.1  # Penalize longer time
        return reward

def greedy_action_selection(state, epsilon, model):
    if random.random() > epsilon:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()  # Greedy action
    else:
        action = random.randint(0, 2)  # Explore: random action
    return action

# Plot UAV path
def plot_uav_paths(uav_path):
    plt.figure(figsize=(8, 8))
    x_vals = [pos[0] for pos in uav_path]
    y_vals = [pos[1] for pos in uav_path]
    plt.plot(x_vals, y_vals, marker="o", linestyle="-", color="b", label="UAV Path")
    plt.scatter(x_vals[0], y_vals[0], color='g', label="Start")
    plt.scatter(x_vals[-1], y_vals[-1], color='r', label="End")
    plt.title("UAV Paths Taken")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = 1.0

    # Training loop
    for episode in range(500):
        x, y = START_X, START_Y  # UAV starting position
        uav_path = [(x, y)]
        total_reward = 0
        time_steps = 0
        done = False

        while not done:
            state = [x, y]
            action = greedy_action_selection(state, epsilon, policy_net)
            
            # Perform action
            if action == 0:  # Move left
                x -= 1
            elif action == 1:  # Move forward
                y += 1
            elif action == 2:  # Move right
                x += 1

            time_steps += 1
            reward = calculate_reward(x, y, END_X, END_Y, time_steps)

            if is_out_of_bounds(x, y):
                reward -= 10  # Extra penalty for going out-of-bounds
                done = True
            elif (x, y) == (END_X, END_Y):
                done = True  # Goal reached

            uav_path.append((x, y))
            total_reward += reward

            next_state = [x, y]
            memory.add((state, action, reward, next_state, done))

            # Experience replay
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                train_batch(batch, policy_net, target_net, optimizer, device)

            if time_steps >= MAX_TIME_STEPS:
                done = True

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Log episode results
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        plot_uav_paths(uav_path)

# Batch training step
def train_batch(batch, policy_net, target_net, optimizer, device):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values for current states
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Loss
    loss = F.mse_loss(q_values.squeeze(), target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Run training
if __name__ == "__main__":
    train()
