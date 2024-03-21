'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''
import random
from enum import Enum

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
    def __init__(self, grid_rows=4, grid_cols=5):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

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

# For unit testing
if __name__=="__main__":
    warehouseRobot = WarehouseRobot()
    warehouseRobot.render()

    for i in range(25):
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        warehouseRobot.perform_action(rand_action)
        warehouseRobot.render()