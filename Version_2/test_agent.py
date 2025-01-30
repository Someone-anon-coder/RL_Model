from Drone_Env import DroneEnv
from Agent import DQNAgent

import time
import pygame

def main():
    env = DroneEnv()
    
    agent = DQNAgent(env)
    # agent.load_agent()
    agent.load_agent("agent_3.pth")

    while True:
        env.reset(obstacle_num=0)
        obstacles = [(env.drone_position[0] + (env.object_dimension * env.scale), env.drone_position[1])]

        done = False

        target_position_set = False
        while not target_position_set:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        state = env.reset(obstacles=obstacles)
                        target_position_set = True
                    
                    elif event.button == 3:
                        x, y = pygame.mouse.get_pos()

                        positions_x = [i for i in range(0, env.screen_width, env.scale * env.object_dimension)]
                        positions_y = [i for i in range(0, env.screen_height, env.scale * env.object_dimension)]

                        x = min(positions_x, key=lambda p: abs(p - x)) - env.scale * env.object_dimension
                        y = min(positions_y, key=lambda p: abs(p - y)) - env.scale * env.object_dimension

                        obstacle = (x, y)
                        obstacles.append(obstacle)

                        state = env.reset(obstacles=obstacles)

            env.render()
            pygame.display.flip()
        
        while not done:
            action = agent.choose_action(state, test=True)
            next_state, _, done, _, results = env.step(action)
            state = next_state

            env.render()
            pygame.display.flip()
            time.sleep(0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            if results[0]:
                print("Drone crashed with wall!")
            if results[1]:
                print("Drone crashed with obstacle!")
            if results[2]:
                print("Drone reached the target!")

            if done:
                print("Episode finished!")

if __name__ == "__main__":
    main()