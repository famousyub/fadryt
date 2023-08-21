import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Custom Environment
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

        # Define the transition dynamics and rewards based on your problem
        # For demonstration purposes, let's assume a simple environment
        # where the goal is to reach state 0
        if self.state == 0 and action == 0:
            next_state = self.state
            reward = 1  # Positive reward for reaching the goal
            done = True  # Episode terminates
        elif self.state == 0 and action != 0:
            next_state = self.state
            reward = -0.1  # Negative reward for taking an action other than the goal
            done = False
        else:
            next_state = np.random.randint(0, self.state_space_size)
            reward = 0
            done = False

        self.state = next_state
        return next_state, reward, done

# Q-learning Agent
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.995):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_space_size, action_space_size))

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action])

        self.exploration_prob *= self.exploration_decay

# Main function
def main():
    state_space_size = 5
    action_space_size = 2
    num_episodes = 1000

    env = CustomEnvironment(state_space_size, action_space_size)
    agent = QLearningAgent(state_space_size, action_space_size)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
