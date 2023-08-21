import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
import time
import copy
import csv
import itertools as it

# Define the action mode enum (assuming you have defined it somewhere)
class ActionMode:
    DIRECT_CHANGE = "DIRECT_CHANGE"

# Define the Config class (assuming you have defined it somewhere)
class Config:
    action_mode = ActionMode.DIRECT_CHANGE

# Define the DQN network
class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))

# Define the ReplayMemory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):
        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size, device):
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer_state)

# Define the DQNAgent class
class DQNAgent:
    def __init__(self, device, state_size, action_size, discount=0.99, eps_max=1.0, eps_min=0.01, eps_decay=0.995,
                 memory_capacity=5000, lr=1e-3, train_mode=True):

        self.device = device
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval()
        if not train_mode:
            self.policy_net.eval()

        self.memory = ReplayMemory(capacity=memory_capacity)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batch_size, self.device)

        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1))
        q_target = self.target_net.forward(next_states).max(dim=1).values
        q_target[dones] = 0.0
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()

    def save_model(self, filename):
        self.policy_net.save_model(filename)

    def load_model(self, filename):
        self.policy_net.load_model(filename=filename, device=self.device)

# Define the remaining utility functions (get_next_state, get_average_latency, get_reward, get_costs_of_paths, get_state_transitions, etc.)


class CustomEnvironment:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.random.randint(0, self.state_space_size)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.action_space_size:
            raise ValueError("Invalid action")

        if self.state == 0 and action == 0:
            next_state = self.state
            reward = 1
            done = True
        elif self.state == 0 and action != 0:
            next_state = self.state
            reward = -0.1
            done = False
        else:
            next_state = np.random.randint(0, self.state_space_size)
            reward = 0
            done = False

        self.state = next_state
        return next_state, reward, done
# Define the main function
if __name__ == "__main__":
    # Initialize the DQNAgent and other parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = 10  # Replace with the appropriate state size
    action_size = 4  # Replace with the appropriate action size
    agent = DQNAgent(device, state_size, action_size)
    state_space_size = 5
    action_space_size = 2
    num_episodes = 1000

    env = CustomEnvironment(state_space_size, action_space_size)

    # Training loop
    for episode in range(1000):  # Replace num_episodes with the desired number of episodes
        state = 1  # Replace initial_state with your initial state
        total_reward = 0

        for step in range(20):  # Replace max_steps_per_episode with the desired number of steps
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)  # Replace env.step() with your environment's step function

            agent.memory.store(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) >= 2:
                agent.learn(2)

            if done:
                break

        agent.update_epsilon()
        agent.update_target_net()

        print(f"Episode {episode + 1} - Reward: {total_reward}")

    # Save the trained model
    agent.save_model("trained_model.pth")
