# RL_Model: Autonomous Drone with Reinforcement Learning

## Description
This project aims to create an autonomous drone using Reinforcement Learning (RL) and Deep Q-Learning (DQN) with a hierarchical agent approach. The project is divided into two versions:

- **Version 1**: Focuses on speed control of the drone to stop at 5 meters from an obstacle.
- **Version 2**: Aims to navigate towards a target while avoiding obstacles.

The goal is to develop a system of agents that control various aspects of the drone to create a fully autonomous flying machine.

## Versions

### Version 1: Speed Control
- **Objective**: The drone learns to control its speed to stop at a specified distance from an obstacle.
- **Key Components**:
  - `Agent.py`: Implements the Q-learning agent.
  - `Drone_Env.py`: Defines the environment in which the drone operates.
  - `test_agent.py`: Contains tests for the Q-learning agent.
  - `test_with_gazebo.py`: Integrates the agent with Gazebo for simulation.

### Version 2: Navigation
- **Objective**: The drone learns to navigate towards a target while avoiding obstacles.
- **Key Components**:
  - `Agent.py`: Implements the DQN agent.
  - `Drone_Env.py`: Defines the environment for navigation.
  - `test_agent.py`: Contains tests for the DQN agent.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RL_Model
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the agents, execute the following commands:

- For Version 1:
  ```bash
  python Version_1/Agent.py
  ```

- For Version 2:
  ```bash
  python Version_2/Agent.py
  ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License
This project is licensed under the MIT License.
