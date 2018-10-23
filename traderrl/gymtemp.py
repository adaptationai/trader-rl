import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Template(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    #Define Actions
    ACTION = [0,1,2]

    def __init__(self, env):
        self.env = env
        self.viewer = None
        self.info = None
        self.reward = None
        self.done = False
        self.state = None

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(3)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(1)
        #print("obs")
        #print (self.observation_space)

        # initial condition
        #self.state = self.env.generate_number()
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        pass

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #self.state = self.env.generate_number()
        #self.env.display()
        self.action = action
        self.reward = self.env.reward(self.action)
        self.done = True
        self.info = None
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.grab_data()
        
        #self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        pass
        return

    def render(self, mode="human", close=False):
        self.env.display()

        return 






