import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
#from game import WhatIsTheColor
#env = WhatIsTheColor()

import numpy as np
from .env2 import Template


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
        self.action_dim = 4
        self.state_dim = 32
        self.num_envs = 1
        self.num_envs_per_sub_batch = 1
        self.total_pips = []
        self.player = self.env.player
        self.pips = self.env.pips
        self.starter = 0
        self.discrete = True

        #self.df = df
        #self.reward_range = (0, MAX_ACCOUNT_BALANCE) 
        if self.discrete:
            # forward or backward in each dimension
            self.action_space = spaces.Discrete(3)
            #self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)

            # observation is the x, y coordinate of the grid
            #low = np.zeros(0, dtype=int)
            #high =  np.array(1, dtype=int) - np.ones(len(self.maze_size), dtype=int)
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(58,))
        else:
            # Actions of the format Buy x%, Sell x%, Hold, etc.
            self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)
            #or
            #self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([3, 1, 1, 1]), dtype=np.float16)


            # Prices contains the OHCL values for the last five prices
            self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

        

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
        #self.placement = self.env.placement
        self.next_state, self.reward, self.done, self.pips = self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        self.total_pips.append(self.pips)
        if self.done:
            #print("total pips")
            #print(np.sum(self.total_pips))
            #print(len(self.total_pips))
            #self.starter += 1
            pass
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


#test = Template_Gym()