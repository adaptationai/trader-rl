
import numpy as np
from state import *

class Template():
    """Base class for Unity ML environments using unityagents (v0.4)."""

    def __init__(self, name, seed=0):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.env = MarketSim()


    def reset(self):
        """Reset the environment."""
        #info = self.env.reset(train_mode=True)[self.brain_name]
        state = self.env.reset()
        return state

    def step(self, action):
        """Take a step in the environment."""
        state, reward, done = self.env.step(action)
        return state, reward, done

    def render(self):
        """
        Render the environment to visualize the agent interacting.
        Does nothing because rendering is always on as is required by linux environments.
        """
        self.env.render()
        self.env.player.render()