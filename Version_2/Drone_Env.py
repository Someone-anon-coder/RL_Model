import random
import pygame
import numpy as np

class DroneEnv():
    def __init__(self) -> None:
        """
            Initialize the drone environment.
        """

        self.screen_width = 600
        self.screen_height = 400
        self.scale = 20 # Each meter is 20 pixels, 2 boxes = 1 meter, if change_scale = 10

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Drone Environment")
        
        self.clock = pygame.time.Clock()

        self.object_dimension = 2
        self.line_thickness = 2

        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.black = (0, 0, 0)
    
    def reset(self, target_position: tuple | None = None, obstacles: list | None = None, obstacle_num=5) -> np.array:
        """
            Set the variables of the environment.

            Args:
                target_position (tuple): The position of the target.
                obstacles (list): The positions of the obstacles. Default is None.
                obstacle_num (int): The number of obstacles. Default is 5.
            
            Returns:
                np.array: The initial state of the environment.
        """
        
        self.drone_position = (self.object_dimension * self.scale // 2 + self.screen_width // 10, self.screen_height // 2)
        self.reward = 0
        
        if target_position is not None:
            self.target_position = target_position
        else:
            # self.target_position = (self.get_target_position(drone_position=self.drone_position), self.screen_height // 2)
            self.target_position = (self.screen_width // 2 + (self.object_dimension // 2 * self.scale), self.screen_height // 2)
        
        self.drone_distance_x = abs(self.drone_position[0] - self.target_position[0])
        self.drone_distance_y = abs(self.drone_position[1] - self.target_position[1])

        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = self.get_obstacles(drone_position=self.drone_position, target_position=self.target_position, obstacles_num=obstacle_num)
        
        (
            if_obstacle_on_left, 
            if_obstacle_on_right, 
            if_obstacle_on_top, 
            if_obstacle_on_bottom,
            if_obstacle_on_left_top,
            if_obstacle_on_right_top,
            if_obstacle_on_left_bottom,
            if_obstacle_on_right_bottom
        ) = self.if_obstacle_on_side(drone_position=self.drone_position, diagonals=True)

        self.reached_target = False
        self.crashed_with_wall = False
        self.crashed_with_obstacle = False

        self.state = np.array([
            self.drone_position[0] / (self.screen_width + self.object_dimension * self.scale),
            self.drone_position[1] / (self.screen_height + self.object_dimension * self.scale),
            self.target_position[0] / (self.screen_width + self.object_dimension * self.scale),
            self.target_position[1] / (self.screen_height + self.object_dimension * self.scale),
            self.drone_distance_x / (self.screen_width // 2 + self.object_dimension * self.scale),
            self.drone_distance_y / (self.screen_height // 2 + self.object_dimension * self.scale),
            if_obstacle_on_left,
            if_obstacle_on_right,
            if_obstacle_on_top,
            if_obstacle_on_bottom,
            if_obstacle_on_left_top,
            if_obstacle_on_right_top,
            if_obstacle_on_left_bottom,
            if_obstacle_on_right_bottom
        ])
        return self.state

    def get_target_position(self, drone_position: tuple) -> tuple:
        """
            Get the target position.

            Args:
                drone_position (tuple): The position of the drone.

            Returns:
                tuple: The target position.
        """

        while True:
            x = random.randint(self.screen_width // 2, self.screen_width - (self.object_dimension * self.scale * 2))
            positions_x = [i for i in range(0, self.screen_width, self.scale * self.object_dimension)]

            x = min(positions_x, key=lambda p: abs(p - x)) + self.scale * self.object_dimension
            if x not in range(drone_position[0], drone_position[0] + self.object_dimension * self.scale):
                return x
    
    def get_obstacles(self, drone_position: tuple, target_position: tuple, obstacles_num: int=5) -> list:
        """
            Get the obstacles in the environment.

            Args:
                drone_position (tuple): The position of the drone.
                target_position (tuple): The position of the target.
                obstacles_num (int): The number of obstacles. Defaults to 5.

            Returns:
                list: The obstacles in the environment.
        """

        obstacles = [(drone_position[0] + (self.object_dimension * self.scale), drone_position[1])]

        for _ in range(0, random.randint(obstacles_num // 2, obstacles_num)):
        # for _ in range(0, obstacles_num):
            x = random.randint(drone_position[0] + (self.object_dimension * self.scale * 2), target_position[0] - (self.object_dimension * self.scale))
            y = random.randint(drone_position[1] - (self.object_dimension * self.scale * 2), drone_position[1] + (self.object_dimension * self.scale * 4))

            positions_x = [i for i in range(drone_position[0] + (self.object_dimension * self.scale * 2), target_position[0], self.scale * self.object_dimension)]
            positions_y = [i for i in range(drone_position[1] - (self.object_dimension * self.scale * 2), drone_position[1] + (self.object_dimension * self.scale * 4), self.scale * self.object_dimension)]
            
            x = min(positions_x, key=lambda p: abs(p - x)) - self.scale * self.object_dimension
            y = min(positions_y, key=lambda p: abs(p - y)) - self.scale * self.object_dimension

            if ((x, y) not in obstacles):
                obstacles.append((x, y))
        
        return obstacles
    
    def if_obstacle_on_side(self, drone_position: tuple, diagonals: bool=False) -> tuple:
        """
            Check if there is an obstacle on the side of the drone.

            Args:
                drone_position (tuple): The position of the drone.
                diagonals (bool): Whether to check diagonals. Defaults to False.

            Returns:
                tuple: If there is an obstacle on the side of the drone.
        """

        if_obstacle_on_left = False
        if_obstacle_on_right = False
        if_obstacle_on_top = False
        if_obstacle_on_bottom = False

        if_obstacle_on_left_top = False
        if_obstacle_on_right_top = False
        if_obstacle_on_left_bottom = False
        if_obstacle_on_right_bottom = False

        for obstacle in self.obstacles:
            if ((drone_position[0] - self.object_dimension * self.scale, drone_position[1]) == obstacle) or (drone_position[0] - self.object_dimension * self.scale < 0):
                if_obstacle_on_left = True
            
            if ((drone_position[0] + self.object_dimension * self.scale, drone_position[1]) == obstacle) or (drone_position[0] + self.object_dimension * self.scale > self.screen_width):
                if_obstacle_on_right = True
            
            if ((drone_position[0], drone_position[1]  - self.object_dimension * self.scale) == obstacle) or (drone_position[1] - self.object_dimension * self.scale < 0):
                if_obstacle_on_top = True
            
            if ((drone_position[0], drone_position[1]  + self.object_dimension * self.scale) == obstacle) or (drone_position[1] + self.object_dimension * self.scale > self.screen_height):
                if_obstacle_on_bottom = True
            
            if diagonals:
                if ((drone_position[0] - (self.object_dimension * self.scale), drone_position[1] - (self.object_dimension * self.scale))) == obstacle:
                    if_obstacle_on_left_top = True
                if ((drone_position[0] + (self.object_dimension * self.scale), drone_position[1] - (self.object_dimension * self.scale))) == obstacle:
                    if_obstacle_on_right_top = True
                if ((drone_position[0] - (self.object_dimension * self.scale), drone_position[1] + (self.object_dimension * self.scale))) == obstacle:
                    if_obstacle_on_left_bottom = True
                if ((drone_position[0] + (self.object_dimension * self.scale), drone_position[1] + (self.object_dimension * self.scale))) == obstacle:
                    if_obstacle_on_right_bottom = True

#         print(f"""
# Obstacle on Left: {if_obstacle_on_left},
# Obstacle on Right: {if_obstacle_on_right}
# Obstacle on Top: {if_obstacle_on_top}
# Obstacle on Bottom: {if_obstacle_on_bottom}
# Obstacle on Left Top: {if_obstacle_on_left_top}
# Obstacle on Right Top: {if_obstacle_on_right_top}
# Obstacle on Left Bottom: {if_obstacle_on_left_bottom}
# Obstacle on Right Bottom: {if_obstacle_on_right_bottom}""")
        
        if diagonals:     
            return (
                    int(if_obstacle_on_left),
                    int(if_obstacle_on_right),
                    int(if_obstacle_on_top),
                    int(if_obstacle_on_bottom),
                    int(if_obstacle_on_left_top),
                    int(if_obstacle_on_right_top),
                    int(if_obstacle_on_left_bottom),
                    int(if_obstacle_on_right_bottom)
                )
        
        return int(if_obstacle_on_left), int(if_obstacle_on_right), int(if_obstacle_on_top), int(if_obstacle_on_bottom)
    
    def step(self, action: int) -> tuple:
        """
            Take a step in the environment.

            Args:
                action (int): The action to be taken by the agent.

            Returns:
                tuple: The next state, reward, done, info of the environment, and step outcome.
        """

        done = False
        previous_drone_position = self.drone_position

        (
            if_obstacle_on_left, 
            if_obstacle_on_right, 
            if_obstacle_on_top, 
            if_obstacle_on_bottom,
            if_obstacle_on_left_top,
            if_obstacle_on_right_top,
            if_obstacle_on_left_bottom,
            if_obstacle_on_right_bottom
        ) = self.if_obstacle_on_side(drone_position=previous_drone_position, diagonals=True)
        
        self.reward += -1
        if action == 0:
            if if_obstacle_on_left:
                print(f"\nDrone crashed with position: ({self.drone_position[0] - self.object_dimension * self.scale}, {self.drone_position[1]})")
                print(f"Obstacles: ", *self.obstacles, sep=" -> ")

                self.crashed_with_obstacle = True
                self.reward += -200
                done = True
            print(f"Moving left with position: {self.drone_position}")
            self.drone_position = (self.drone_position[0] - self.object_dimension * self.scale, self.drone_position[1])
            self.drone_distance_x = abs(self.drone_position[0] - self.target_position[0])
        
        elif action == 1:
            if if_obstacle_on_right:
                print(f"\nDrone crashed with position: ({self.drone_position[0] + self.object_dimension * self.scale}, {self.drone_position[1]})")
                print(f"Obstacles: ", *self.obstacles, sep=" -> ")

                self.crashed_with_obstacle = True
                self.reward += -200
                done = True
            print(f"Moving right with position: {self.drone_position}")
            self.drone_position = (self.drone_position[0] + self.object_dimension * self.scale, self.drone_position[1])
            self.drone_distance_x = abs(self.drone_position[0] - self.target_position[0])
        
        elif action == 2:
            if if_obstacle_on_top:
                print(f"\nDrone crashed with position: ({self.drone_position[0]}, {self.drone_position[1] - self.object_dimension * self.scale})")
                print(f"Obstacles: ", *self.obstacles, sep=" -> ")

                self.crashed_with_obstacle = True
                self.reward += -200
                done = True
            print(f"Moving top with position: {self.drone_position}")
            self.drone_position = (self.drone_position[0], self.drone_position[1] - self.object_dimension * self.scale)
            self.drone_distance_y = abs(self.drone_position[1] - self.target_position[1])
        
        elif action == 3:
            if if_obstacle_on_bottom:
                print(f"\nDrone crashed with position: ({self.drone_position[0]}, {self.drone_position[1] + self.object_dimension * self.scale})")
                print(f"Obstacles: ", *self.obstacles, sep=" -> ")
                
                self.crashed_with_obstacle = True
                self.reward += -200
                done = True
            print(f"Moving bottom with position: {self.drone_position}")
            self.drone_position = (self.drone_position[0], self.drone_position[1] + self.object_dimension * self.scale)
            self.drone_distance_y = abs(self.drone_position[1] - self.target_position[1])
        
        if self.drone_position == self.target_position or (self.drone_position[1] == self.target_position[1] and self.drone_position[0] >= self.target_position[0]):
            print("\n\nDrone Reached the target")
            
            self.reached_target = True
            self.reward += 100
            done = True
        
        if self.drone_position[0] < 0 or self.drone_position[0] > self.screen_width or self.drone_position[1] < 0 or self.drone_position[1] > self.screen_height:
            print(f"\nDrone Crashed with wall with position: {self.drone_position}")
            
            self.crashed_with_wall = True
            self.reward += -200
            done = True
        
        if (abs(self.drone_position[0] - self.target_position[0]) < abs(previous_drone_position[0] - self.target_position[0])) or (abs(self.drone_position[1] - self.target_position[1]) < abs(previous_drone_position[1] - self.target_position[1])):
            self.reward += 5
        elif (abs(self.drone_position[0] - self.target_position[0]) > abs(previous_drone_position[0] - self.target_position[0])) or (abs(self.drone_position[1] - self.target_position[1]) > abs(previous_drone_position[1] - self.target_position[1])):
            self.reward += -5

        (
            if_obstacle_on_left, 
            if_obstacle_on_right, 
            if_obstacle_on_top, 
            if_obstacle_on_bottom,
            if_obstacle_on_left_top,
            if_obstacle_on_right_top,
            if_obstacle_on_left_bottom,
            if_obstacle_on_right_bottom
        ) = self.if_obstacle_on_side(drone_position=self.drone_position, diagonals=True)
        
        self.state = np.array([
            self.drone_position[0] / (self.screen_width + self.object_dimension * self.scale),
            self.drone_position[1] / (self.screen_height + self.object_dimension * self.scale),
            self.target_position[0] / (self.screen_width + self.object_dimension * self.scale),
            self.target_position[1] / (self.screen_height + self.object_dimension * self.scale),
            self.drone_distance_x / (self.screen_width // 2 + self.object_dimension * self.scale),
            self.drone_distance_y / (self.screen_height // 2 + self.object_dimension * self.scale),
            if_obstacle_on_left,
            if_obstacle_on_right,
            if_obstacle_on_top,
            if_obstacle_on_bottom,
            if_obstacle_on_left_top,
            if_obstacle_on_right_top,
            if_obstacle_on_left_bottom,
            if_obstacle_on_right_bottom
        ])
        
        print(f"\nDrone Next position: {self.drone_position}")
        print(f"Target Next position: {self.target_position}")
        print(f"Next State: {self.state}")
        print(f"Reward: {self.reward}\n")

        print("<--", "*"*50, "-->")

        return self.state, self.reward, done, {}, (self.crashed_with_wall, self.crashed_with_obstacle, self.reached_target)

    def render(self) -> None:
        """
            Render the environment to the screen.

            Args:
                target_position (tuple): The position of the target.
        """

        self.screen.fill((255, 255, 255))

        pygame.draw.rect(self.screen, self.green, (self.drone_position[0], self.drone_position[1], self.object_dimension * self.scale, self.object_dimension * self.scale))
        if self.target_position is not None:
            pygame.draw.rect(self.screen, self.red, (self.target_position[0], self.target_position[1], self.object_dimension * self.scale, self.object_dimension * self.scale))

        # # Next state of drone to check collision
        # pygame.draw.rect(self.screen, self.blue, (self.drone_position[0] - self.object_dimension * self.scale, self.drone_position[1], self.object_dimension * self.scale, self.object_dimension * self.scale))
        # pygame.draw.rect(self.screen, self.blue, (self.drone_position[0] + self.object_dimension * self.scale, self.drone_position[1], self.object_dimension * self.scale, self.object_dimension * self.scale))
        # pygame.draw.rect(self.screen, self.blue, (self.drone_position[0], self.drone_position[1] - self.object_dimension * self.scale, self.object_dimension * self.scale, self.object_dimension * self.scale))
        # pygame.draw.rect(self.screen, self.blue, (self.drone_position[0], self.drone_position[1] + self.object_dimension * self.scale, self.object_dimension * self.scale, self.object_dimension * self.scale))

        # Draw Grid
        for x in range(0, self.screen_width - self.scale, self.object_dimension * self.scale):
            pygame.draw.line(self.screen, self.black, (x, 0), (x, self.screen_height), self.line_thickness)
        for y in range(0, self.screen_height - self.scale, self.object_dimension * self.scale):
            pygame.draw.line(self.screen, self.black, (0, y), (self.screen_width, y), self.line_thickness)

        # Draw obstacles
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                pygame.draw.rect(self.screen, self.black, (obstacle[0], obstacle[1], self.object_dimension * self.scale, self.object_dimension * self.scale))        

if __name__ == "__main__":
    env = DroneEnv()
    env.reset(obstacle_num=10)

    while True:
        env.render()
        pygame.display.flip()
        
        env.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()