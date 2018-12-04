import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
#from game import WhatIsTheColor
#env = WhatIsTheColor()

import numpy as np
from .env import Template


class Template_Gym(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    #Define Actions
    ACTION = [0,1]

    def __init__(self, env=Template()):
        self.env = env
        self.viewer = None
        self.info = None
        self.reward = None
        self.done = False
        self.state = None
        self.action_dim = 3
        self.state_dim = 366
        self.num_envs = 1
        self.num_envs_per_sub_batch = 1


        # forward or backward in each dimension
        self.action_space = spaces.Discrete(3)

        # observation is the x, y coordinate of the grid
        #low = np.zeros(0, dtype=int)
        #high =  np.array(1, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(366,))
        #print("obs")
        #print (self.observation_space)

        # initial condition
        #self.state = self.env.generate_number()
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        #self.reset()

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
        #print(action)
        self.next_state, self.reward, self.done = self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        return self.next_state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.reset()
        #self.reward = np.array([reward])
        #self.state = self.state.tolist()
        #self.state = np.array([self.state])
        #self.steps_beyond_done = None
        self.done = False
        #self.done = np.array([self.done])
        return self.state

    def is_game_over(self):
        pass
        return

    def render(self, mode="human", close=False):
        #self.env.display()
        pass

        return 


test = Template_Gym()