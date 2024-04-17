'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_warehouse_robot as wr
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='warehouse-robot-v0',                                # call it whatever you want
    entry_point='v0_warehouse_robot_env:WarehouseRobotEnv', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class WarehouseRobotEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, grid_rows=4, grid_cols=5, render_mode=None):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        # Initialize the WarehouseRobot problem
        self.warehouse_robot = wr.WarehouseRobot(grid_rows=grid_rows, grid_cols=grid_cols, fps=self.metadata['render_fps'])

        # Gym requires defining the action space. The action space is robot's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Gym requires defining the observation space. The observation space consists of the robot's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]),
            shape=(4,),
            dtype=np.int32
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
        self.warehouse_robot.reset(seed=seed)

        # Construct the observation state:
        # [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
        obs = np.concatenate((self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos))
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        target_reached = self.warehouse_robot.perform_action(wr.RobotAction(action))

        # Determine reward and termination
        reward=0
        terminated=False
        if target_reached:
            reward=1
            terminated=True

        # Construct the observation state: 
        # [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
        obs = np.concatenate((self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos))

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            print(wr.RobotAction(action))
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info

    # Gym required function to render environment
    def render(self):
        self.warehouse_robot.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('warehouse-robot-v0', render_mode='human')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]
