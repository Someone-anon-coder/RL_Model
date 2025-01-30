from sys import argv
import time
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Drone_Env import DroneEnv

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
            Define a simple neural network for DQN.

            Args:
                input_dim: The number of input features (state dimension).
                output_dim: The number of actions (action dimension).
        """

        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: The input tensor.

            Returns:
                The output tensor.
        """

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return self.fc4(x)

class DQNAgent:
    def __init__(self, env: DroneEnv, alpha: float=0.0001, gamma: float=0.95, epsilon: float=0.9, epsilon_min: float=0.05, epsilon_decay: float | None = None, batch_size: int=32, memory_size: int=10000) -> None:
        """
            Initialize the DQN agent.

            Args:
                env: The environment the agent interacts with.
                alpha: Learning rate for the optimizer.
                gamma: Discount factor.
                epsilon: Exploration rate.
                batch_size: Mini-batch size for training.
                memory_size: Maximum size of the experience replay buffer.
        """
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.state_dim = 14 # Number of inputs (states) for the agent (drone_position, target_position, drone_distance, if_obstacles_on_side) 
        self.action_dim = 4 # Number of possible actions (left, right, up, down)
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        
    def choose_action(self, state: np.array, test: bool=False, first_some_episodes: bool=False) -> int:
        """
            Choose an action using an epsilon-greedy policy.

            Args:
                state: Current state.
                test: Whether the agent is in test mode. Defaults to False.
                first_some_episodes: Whether the agent is in the first some episodes. Defaults to False.

            Returns:
                action: Chosen action.
        """
        
        if test:
            self.epsilon = 0

        if first_some_episodes:
            return random.choice(range(self.action_dim))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))  # Exploration: random action
        
        else:
            if self.epsilon_decay:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
        
            print(f"Actions (Q-values): {q_values}")
            return torch.argmax(q_values).item()  # Exploitation: action with highest Q-value
    
    def store_experience(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        """
            Store an experience in the memory buffer.

            Args:
                state: Current state.
                action: Action taken.
                reward: Reward received.
                next_state: Next state after action.
                done: Whether the episode is done.
        """

        self.memory.append((state, action, reward, next_state, done))
    
    def sample_experience(self) -> list:
        """
            Sample a mini-batch from the experience replay buffer.

            Returns:
                A mini-batch of experiences.
        """
        
        return random.sample(self.memory, self.batch_size)
    
    def update_q_values(self) -> None:
        """
            Update the Q-network using a mini-batch from the experience replay buffer.
        """
        
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in the buffer
        
        batch = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1, keepdim=True)[0]

        target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self, tau: float=0.001) -> None:
        """
            Update the target network using the Q-network.

            Args:
                tau: Interpolation factor for updating the target network. Defaults to 0.001.
        """

        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def train(self, episodes: int = 1000, target_update_freq: int = 10, render:bool = False) -> None:
        """
            Train the DQN agent by interacting with the environment.

            Args:
                episodes: Number of episodes to train the agent. Defaults to 1000.
                target_update_freq: Frequency of updating the target network. Defaults to 10.
                render: Whether to render the environment during training. Defaults to False.
        """
        
        drone_crashed_with_wall = 0
        drone_crashed_with_obstacle = 0
        drone_reached_target = 0

        for episode in range(episodes):
            state = self.env.reset(target_position=None, obstacles=None, obstacle_num=15)
            
            done = False
            total_reward = 0

            while not done:
                if pygame.event.get(pygame.QUIT):
                    pygame.quit()
                    quit()

                print("<--", "*"*50, "-->\n")
                print(f"Episode: {episode}, Epsilon: {self.epsilon}")
                print(f"Drone Current State: {state}")
                if episode < (episode * 0.1):
                    action = self.choose_action(state, first_some_episodes=True)
                else:
                    action = self.choose_action(state)
                
                next_state, reward, done, _, results = self.env.step(action)

                self.store_experience(state, action, reward, next_state, done)
                self.update_q_values()

                state = next_state
                total_reward += reward

                if render:
                    self.env.render()
                    pygame.display.flip()
                    time.sleep(0.5)

            if episode % target_update_freq == 0:
                self.update_target_network()

            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}")

            if results[0]:
                drone_crashed_with_wall += 1
            if results[1]:
                drone_crashed_with_obstacle += 1
            if results[2]:
                drone_reached_target += 1

        print(f"Drone crash count with wall: {drone_crashed_with_wall}")
        print(f"Drone crash count with obstacle: {drone_crashed_with_obstacle}")
        print(f"Drone reached target count: {drone_reached_target}")

    def save_agent(self, filename: str = "dqn_agent.pth") -> None:
        """
            Save the agent's Q-network and target network to a file.

            Args:
                filename (str): The path to the file where the agent will be saved. Defaults to "dqn_agent.pth".
        """
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
        print(f"Agent saved to {filename}")
    
    def load_agent(self, filename: str = "dqn_agent.pth") -> None:
        """
            Load the agent's Q-network and target network from a file.

            Args:
                filename (str): The path to the file where the agent is saved. Defaults to "dqn_agent.pth".
        """
        
        checkpoint = torch.load(filename)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        
        print(f"Q-network and target network loaded from {filename}")

if __name__ == "__main__":
    env = DroneEnv()
    agent = DQNAgent(env, alpha=0.0001, gamma=0.95, epsilon=0.99, epsilon_min=0.1, epsilon_decay=0.99995)
    
    # agent.train(episodes=20000, target_update_freq=100, render=True, render_freq=500)
    render = argv[1] if len(argv) > 1 else False
    agent.train(episodes=20000, target_update_freq=100, render=render)
    agent.save_agent("agent_3.pth")