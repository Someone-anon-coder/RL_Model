import numpy as np
import pygame

class DroneEnv():
    def __init__(self) -> None:
        """
            Initialize the drone environment.
        """

        self.screen_width = 1500
        self.screen_height = 500
        self.scale = 20 # Each meter is 20 pixels, 2 boxes = 1 meter, if change_scale = 10

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Drone Environment")
        
        self.clock = pygame.time.Clock()

        self.drone_dimension = 2
        self.target_dimension = 2
        self.line_thickness = 2

        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.orange = (255, 165, 0)
        self.yellow = (255, 255, 0)
        self.black = (0, 0, 0)
    
    def set_variables(self, target_position: tuple | None = None) -> None:
        """
            Set the variables of the environment.

            Args:
                target_position (tuple): The position of the target.
        """
        
        self.drone_position = (self.screen_width // 10, self.screen_height // 2)
        
        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = (self.screen_width + 1, self.screen_height // 2)

        self.distance_5_meter_position = (self.target_position[0] - (5 * self.scale), self.target_position[0] - (5 * self.scale))
        self.distance_7_meter_position = (self.target_position[0] - (7 * self.scale), self.target_position[0] - (7 * self.scale))
        self.distance_30_meter_position = (self.drone_position[0] + (30 * self.scale), self.drone_position[0] + (30 * self.scale))

        self.drone_speed = 1
        self.max_speed = 5
        
        self.drone_distance = abs(self.target_position[0] - (self.drone_position[0] + self.drone_dimension)) // self.scale
        self.max_distance = 30

        self.target_distance_range = (0, 5)
        self.speed_step = 1

        print(f"Drone position: {self.drone_position[0]}")
        print(f"Target position: {self.target_position[0]}")
        print(f"Drone Distance: {self.drone_distance}")

        self.state = np.array([self.drone_speed, self.drone_distance if self.drone_distance < 31 else 31])

    def reset(self, target_position: tuple | None = None) -> np.array:
        """
            Reset the environment to the initial state.

            Returns:
                np.array: The initial state of the environment.
        """

        self.drone_speed = 1
        self.speed_step = 1

        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = (self.screen_width + 1, self.screen_height // 2)

        self.drone_distance = abs(self.target_position[0] - (self.drone_position[0] + self.drone_dimension)) // self.scale

        self.distance_5_meter_position = (self.target_position[0] - (5 * self.scale), self.target_position[0] - (5 * self.scale))
        self.distance_7_meter_position = (self.target_position[0] - (7 * self.scale), self.target_position[0] - (7 * self.scale))
        self.distance_30_meter_position = (self.drone_position[0] + (30 * self.scale), self.drone_position[0] + (30 * self.scale))
        
        self.state = np.array([self.drone_speed, self.drone_distance if self.drone_distance < 31 else 31])  # Reset to initial speed and max distance
        
        return self.state
    
    def step(self, action: int) -> tuple:
        """
            Take a step in the environment.

            Args:
                action (int): The action to be taken by the agent.

            Returns:
                tuple: The next state, reward, done, and info of the environment.
        """

        done = False

        if self.target_position[0] <= self.distance_30_meter_position[0]:
            if action == 0:
                self.drone_speed = min(self.drone_speed + 1, self.max_speed)
                self.speed_step += 1
            
            elif action == 1:
                self.drone_speed = max(self.drone_speed - 1, 0)
                self.speed_step += 1
            
            elif action == 2:
                self.speed_step += 1
        
        if self.drone_distance < 31:
            self.drone_distance -= self.drone_speed
        else:
            self.drone_distance -= 1
        
        self.target_position = (self.target_position[0] - self.drone_speed * self.scale, self.target_position[1])
        self.state = np.array([self.drone_speed, self.drone_distance if self.drone_distance < 31 else 31])

        if self.target_distance_range[0] <= self.drone_distance <= self.target_distance_range[1]:
            reward = 10 // self.speed_step
            done = True
        
        elif 0 <= self.drone_distance < self.target_distance_range[1]:
            reward = -10
            done = True
        
        else:
            reward = -1

        print(f"Drone position: {self.drone_position[0]}")
        print(f"Target position: {self.target_position[0]}")
        print(f"Drone Distance: {self.drone_distance}")

        return self.state, reward, done, {}

    def render(self) -> None:
        """
            Render the environment to the screen.

            Args:
                target_position (tuple): The position of the target.
        """

        self.screen.fill((255, 255, 255))

        pygame.draw.rect(self.screen, self.green, (self.drone_position[0] - self.drone_dimension * 10, self.drone_position[1] - self.drone_dimension * 4, self.drone_dimension * self.scale, self.drone_dimension * self.scale))
        pygame.draw.rect(self.screen, self.red, (self.target_position[0] - self.target_dimension * 10, self.target_position[1] - self.target_dimension * 4, self.target_dimension * self.scale, self.target_dimension * self.scale))

        change_scale = 10

        # Draw Grid
        for i in range(int(self.screen_height // change_scale)):
            pygame.draw.line(self.screen, self.black, (0, i * change_scale), (self.screen_width, i * change_scale), self.line_thickness)
        
        for i in range(int(self.screen_width // change_scale)):
            pygame.draw.line(self.screen, self.black, (i * change_scale, 0), (i * change_scale, self.screen_height), self.line_thickness)

        self.distance_5_meter_position = (self.target_position[0] - (5 * self.scale), self.target_position[0] - (5 * self.scale))
        self.distance_7_meter_position = (self.target_position[0] - (7 * self.scale), self.target_position[0] - (7 * self.scale))
        self.distance_30_meter_position = (self.drone_position[0] + (30 * self.scale), self.drone_position[0] + (30 * self.scale))

        # 5 meter Distance from object
        pygame.draw.line(self.screen, self.blue, (self.distance_5_meter_position[0], 0), (self.distance_5_meter_position[1], self.screen_height), self.line_thickness * 2)
        pygame.draw.line(self.screen, self.blue, (self.distance_5_meter_position[0] + 2, 0), (self.distance_5_meter_position[1] + 2, self.screen_height), self.line_thickness * 2)

        # 7 meter Distance from object
        pygame.draw.line(self.screen, self.orange, (self.distance_7_meter_position[0], 0), (self.distance_7_meter_position[1], self.screen_height), self.line_thickness * 2)
        pygame.draw.line(self.screen, self.orange, (self.distance_7_meter_position[0] + 2, 0), (self.distance_7_meter_position[1] + 2, self.screen_height), self.line_thickness * 2)

        # 30 meter Distance from Drone
        pygame.draw.line(self.screen, self.yellow, (self.distance_30_meter_position[0], 0), (self.distance_30_meter_position[1], self.screen_height), self.line_thickness * 2)
        pygame.draw.line(self.screen, self.yellow, (self.distance_30_meter_position[0] + 2, 0), (self.distance_30_meter_position[1] + 2, self.screen_height), self.line_thickness * 2)

if __name__ == "__main__":
    env = DroneEnv()

    env.set_variables(target_position=(5 * env.screen_width // 6, env.screen_height // 2))
    while True:
        env.render()
        pygame.display.flip()
        
        env.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()