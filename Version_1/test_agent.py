import pygame

from Drone_Env import DroneEnv
from Agent import QLearningAgent

def test_agent() -> None:
    env = DroneEnv()
    agent = QLearningAgent(env)
    agent.load_agent()

    while True:
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
            action = agent.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state

            env.render()
            pygame.display.flip()
            pygame.time.Clock().tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

if __name__ == "__main__":
    test_agent()