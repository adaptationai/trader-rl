
import numpy as np
from .state import *

class Template():
    """Base class for Unity ML environments using unityagents (v0.4)."""

    def __init__(self, name='trader', seed=0):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.start = 0
        self.env = MarketSim(self.start)
        self.player = self.env.player
        self.pips = self.env.pips
        


    def reset(self):
        """Reset the environment."""
        #info = self.env.reset(train_mode=True)[self.brain_name]
        self.env = MarketSim(self.start)
        state = self.env.reset()
        return state

    def step(self, action):
        """Take a step in the environment."""
        #self.placement = self.env.placement
        state, reward, done, self.pips = self.env.step(action)
        #print(done)
        if done:
            self.start += 1

        return state, reward, done, self.pips

    def render(self):
        """
        Render the environment to visualize the agent interacting.
        Does nothing because rendering is always on as is required by linux environments.
        """
        self.env.render()
        self.env.player.render()

#test = Template("trader-rl")
#state = test.reset()
#print(len(state))
#print(test.env.state[0])
#k = test.state_maker()
#print
#for step in range(len(test.env.state)):
    #test.step(1440)