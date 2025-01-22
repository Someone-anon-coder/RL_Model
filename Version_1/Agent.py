import time
import random
import pygame
import pickle
import numpy as np
from Drone_Env import DroneEnv

class QLearningAgent:
    def __init__(self, env: DroneEnv, alpha: float=0.1, gamma: float=0.9, epsilon: float=0.2) -> None:
        """
            Initialize the Q-learning agent.

            Args:
                env: The environment that the agent will interact with.
                alpha: Learning rate
                gamma: Discount factor
                epsilon: Exploration rate
        """

        self.env = env
        self.env.set_variables(target_position = (5 * self.env.screen_width // 6, self.env.screen_height // 2))
        
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        self.q_table = {}
        
    def get_state_key(self, state: tuple) -> tuple:
        """
            Convert state (drone_speed, drone_distance) into a tuple for Q-table indexing.

            Args:
                state (tuple): The current state of the environment.
        """
        
        speed, distance = state
        return (speed, distance)

    def initialize_q_table(self) -> None:
        """
            Initialize Q-table with zeros for all possible state-action pairs.
        """
        
        for speed in range(6):
            for distance in range(0, 32): # 0 to 31, 31 is the distance when target is not visible
                self.q_table[(speed, distance)] = [0.0, 0.0, 0.0]
    
    def choose_action(self, state: tuple) -> int:
        """
            Choose an action based on epsilon-greedy strategy.
        
            Args:
                state: Current state (drone_speed, drone_distance).
        
            Returns:
                action: The action to take (0: Increase, 1: Decrease, 2: Constant).
        """

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2])  # Exploration: random action
        
        else:
            speed, distance = state
            return np.argmax(self.q_table[(speed, distance)])  # Exploitation: best action based on Q-values
    
    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """
            Update Q-value based on the Q-learning update rule.
            
            Args:
                state: Current state (drone_speed, drone_distance).
                action: The action taken by the agent.
                reward: The reward received from the environment.
                next_state: The next state (drone_speed, drone_distance).
        """
        
        speed, distance = state
        next_speed, next_distance = next_state

        max_future_q = np.max(self.q_table[(next_speed, next_distance)])
        current_q = self.q_table[(speed, distance)][action]
        
        # Q-learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(speed, distance)][action] = new_q

    def save_agent(self, filename: str = "agent.pkl") -> None:
        """
            Save the agent's Q-table to a file using pickle.

            Args:
                filename (str): The path to the file where the Q-table will be saved. Default is "agent.pkl".
        """

        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Agent saved to {filename}")
    
    def load_agent(self, filename: str = "agent.pkl") -> None:
        """
            Load the agent's Q-table from a file using pickle.

            Args:
                filename (str): The path to the file where the Q-table is saved. Default is "agent.pkl".
        """

        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Agent loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty Q-table.")
            self.initialize_q_table()
    
    def train(self, episodes: int=1000, delay: int=500) -> None:
        """
            Train the Q-learning agent by interacting with the environment.

            Args:
                episodes: Number of episodes to train the agent.
                delay: Delay between episodes (in milliseconds).
        """
        
        self.initialize_q_table()

        for episode in range(episodes):
            print(f"Episode {episode}/{episodes}")

            state = self.env.reset(target_position=(5 * self.env.screen_width // 6, self.env.screen_height // 2))
            done = False
            total_reward = 0

            while not done:
                if pygame.event.get(pygame.QUIT):
                    pygame.quit()
                    quit()

                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

                self.env.render()
                pygame.display.flip()
                
                time.sleep(0.1)
                # pygame.time.wait(3000)

            # pygame.time.wait(3000)
            pygame.time.wait(delay)
            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    env = DroneEnv()
    
    agent = QLearningAgent(env)
    agent.train(episodes=1000, delay=500)

    agent.save_agent("agent.pkl")