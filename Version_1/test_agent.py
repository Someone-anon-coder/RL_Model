import numpy
import pygame
import random

from Drone_Env import DroneEnv
from Agent import QLearningAgent

def test_agent() -> None:
    env = DroneEnv()
    env.reset()

    agent = QLearningAgent(env)
    agent.load_agent("agent_5.pkl")
    # agent.load_agent()

    while True:
        print("\n\nStarting Testing...\n\n")
        done = False

        target_position_set = False
        while not target_position_set:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, _ = pygame.mouse.get_pos()
                    
                    positions = [i for i in range(0, env.screen_width, env.scale)]
                    x = min(positions, key=lambda p: abs(p - x)) + env.scale // 2 # Round to the nearest grid position

                    target_position = (x, env.screen_height // 2)
                    
                    state = env.reset(target_position=target_position)
                    target_position_set = True

            env.render()
            pygame.display.flip()
            # pygame.time.Clock().tick(60)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, _ = pygame.mouse.get_pos()
                    
                    positions = [i for i in range(0, env.screen_width, env.scale)]
                    x = min(positions, key=lambda p: abs(p - x)) + env.scale // 2 # Round to the nearest grid position

                    target_position = (x, env.screen_height // 2)
                    
                    state = env.reset(target_position=target_position)
                    # target_position_set = True

            action = agent.choose_action(state, test=True)
            next_state, _, done, _ = env.step(action)
            state = next_state

            env.render()
            pygame.display.flip()
            pygame.time.Clock().tick(60)

if __name__ == "__main__":
    test_agent()
    # env = DroneEnv()
    # env.reset(target_position=(5 * env.screen_width // 6, env.screen_height // 2))
    
    # agent = QLearningAgent(env)
    # agent.load_agent("agent_2.pkl")

    # print(f"\n\nTesting...")
    # for _ in range(20):
    #     state = numpy.array([random.randint(0, 5), random.randint(0, 31)])
    #     action = agent.choose_action(state, test=True)

    #     print(f"Speed: {state[0]} | Distance: {state[1]}")
    #     print(f"{'Increase Speed' if action == 0 else 'Decrease Speed' if action == 1 else 'Maintain Speed'}")