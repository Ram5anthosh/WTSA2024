import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Hyperparameters
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_TIME_STEPS = 200
START_X, START_Y = 38, 17  # UAV start coordinates
END_X, END_Y = 43, 97  # UAV end coordinates

# SkyScenes dataset path
DATASET_PATH = './SkyScenes'

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert images to tensors
])

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Adding convolutional layers for image inputs
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        # Fully connected layers for decision making based on image features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 actions: left, forward, right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def calculate_reward(x, y, end_x, end_y, time_steps):
    distance_to_goal = np.sqrt((x - end_x) ** 2 + (y - end_y) ** 2)
    reward = -distance_to_goal - time_steps * 0.01  # Negative reward for time steps
    if (x, y) == (end_x, end_y):
        reward += 10  # Positive reward for reaching the goal
    return reward

def is_out_of_bounds(x, y):
    return x < 38 or x > 43 or y < 17 or y > 97  # Bounds for the environment

def greedy_action_selection(state, epsilon, model, device):
    if random.random() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()  # Greedy action
    else:
        action = random.randint(0, 2)  # Explore: random action
    return action

def load_image(x, y):
    # Load the corresponding image from SkyScenes dataset based on the UAV's position
    folder_path = os.path.join(DATASET_PATH, f'H_15_P_0/ClearNoon/Town07')
    image_path = os.path.join(folder_path, f'00{x}{y}_clrnoon.png')
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return transform(image)  # Apply transformations (resize, etc.)
    else:
        return None  # Image not found (out of bounds)

def train_batch(batch, policy_net, target_net, optimizer, device):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = 1.0
    all_uav_paths = []  # Store paths for plotting

    # Training loop
    for episode in range(500):
        x, y = START_X, START_Y  # UAV starting position
        uav_path = [(x, y)]
        total_reward = 0
        time_steps = 0
        done = False

        while not done:
            image = load_image(x, y)
            if image is None:
                # Out of bounds, end episode
                done = True
                reward = -10  # Penalty for out-of-bounds
                next_state = torch.zeros((3, 64, 64))  # Placeholder for out-of-bounds
            else:
                state = image.unsqueeze(0).to(device)  # Add batch dimension
                action = greedy_action_selection(state, epsilon, policy_net, device)

                # Perform action
                if action == 0:  # Move left
                    y += 10
                elif action == 1:  # Move forward
                    x += 1
                    y += 10
                elif action == 2:  # Move right
                    x += 1

                time_steps += 1
                reward = calculate_reward(x, y, END_X, END_Y, time_steps)

                if is_out_of_bounds(x, y):
                    reward -= 10  # Extra penalty for going out-of-bounds
                    done = True
                elif (x, y) == (END_X, END_Y):
                    reward += 10
                    done = True  # Goal reached

                uav_path.append((x, y))
                total_reward += reward

                next_image = load_image(x, y)
                if next_image is None:
                    next_state = torch.zeros((3, 64, 64))  # Placeholder for out-of-bounds
                else:
                    next_state = next_image.unsqueeze(0).to(device)

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
        all_uav_paths.append(uav_path)

    # Plot all UAV paths (visualization)
    plt.figure(figsize=(10, 10))
    for path in all_uav_paths:
        x_coords, y_coords = zip(*path)
        plt.plot(x_coords, y_coords, marker='o')
    plt.scatter([START_X], [START_Y], color='green', marker='X', s=100, label='Start')
    plt.scatter([END_X], [END_Y], color='red', marker='X', s=100, label='End')
    plt.legend()
    plt.grid(True)
    plt.show()

train()
