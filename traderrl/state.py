import numpy as numpy
from utilities import DataGrabber
import torch

class MarketSim():
    def __init__(self, player):
        self.love = 14
        self.player = player
        self.actions = [1,2,3,4]
        self.years = ['2000', '2001', '2002']
        self.count = [1440, 2880]
        self.granularity= ['M1', 'M5', 'M15', 'M30', 'H1', 'H4'] 
        self.instrument = ['EUR_USD']
        self.data_grabber = DataGrabber()
        self.time = ['00:00:00']
        self.day = ['01']
        self.month = ['01']
        self.state = None

    
    def make_episode(self):
        self.state = self.data_grabber.process_to_normalized()


    def make_current_state(self, count):
        start = (0+count)
        end = (1440+count)
        state = self.state[start:end]
        return state

    def step(self):
        state = self.state
        
        return state


    def reset(self):
        pass

