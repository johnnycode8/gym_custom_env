'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''
import random
from enum import Enum
import pygame
import sys
from os import path

# Actions the Robot is capable of performing i.e. go in a certain direction
class RobotAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3

# The Warehouse is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    _FLOOR=0
    ROBOT=1
    TARGET=2

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        return self.name[:1]

class WarehouseRobot:

    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, grid_rows=4, grid_cols=5, fps=1):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

        self.fps = fps
        self.last_action=''
        self._init_pygame()

    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/package.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size) 


    def reset(self, seed=None):
        # Initialize Robot's starting position
        self.robot_pos = [0,0]

        # Random Target position
        random.seed(seed)
        self.target_pos = [
            random.randint(1, self.grid_rows-1),
            random.randint(1, self.grid_cols-1)
        ]

    def perform_action(self, robot_action:RobotAction) -> bool:
        self.last_action = robot_action

        # Move Robot to the next cell
        if robot_action == RobotAction.LEFT:
            if self.robot_pos[1]>0:
                self.robot_pos[1]-=1
        elif robot_action == RobotAction.RIGHT:
            if self.robot_pos[1]<self.grid_cols-1:
                self.robot_pos[1]+=1
        elif robot_action == RobotAction.UP:
            if self.robot_pos[0]>0:
                self.robot_pos[0]-=1
        elif robot_action == RobotAction.DOWN:
            if self.robot_pos[0]<self.grid_rows-1:
                self.robot_pos[0]+=1

        # Return true if Robot reaches Target
        return self.robot_pos == self.target_pos

    def render(self):
        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):

                if([r,c] == self.robot_pos):
                    print(GridTile.ROBOT, end=' ')
                elif([r,c] == self.target_pos):
                    print(GridTile.TARGET, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print() # new line
        print() # new line

        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255,255,255))

        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if([r,c] == self.target_pos):
                    # Draw target
                    self.window_surface.blit(self.goal_img, pos)

                if([r,c] == self.robot_pos):
                    # Draw robot
                    self.window_surface.blit(self.robot_img, pos)
                
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)  

    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                


# For unit testing
if __name__=="__main__":
    warehouseRobot = WarehouseRobot()
    warehouseRobot.render()

    while(True):
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        warehouseRobot.perform_action(rand_action)
        warehouseRobot.render()