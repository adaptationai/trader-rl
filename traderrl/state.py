import numpy as numpy
from utilities import DataGrabber
import torch
from player import Player

class MarketSim():
    def __init__(self):
        self.love = 14
        self.player = Player()
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
        self.state_full = None
        self.state_current = None
        self.price = None
        self.count = 0
    
    def make_episode(self):
        self.state_full = self.data_grabber.process_to_array()


    def make_current_state(self, count):
        start = (0+count)
        end = (1440+count)
        self.state = self.state_full[start:end]
        return self.state

    def get_price(self):
        self.state_current = self.state[-1:]
        self.price = self.state_current[0]

    def step(self, action):
        self.state = self.make_current_state()
        
        action = action
        reward = self.env.reward(self.action)
        done = True
        info = None
        self.count += 1
        return self.state, self.reward, self.done, self.info


    def reset(self):
        self.count = 0
        self.make_episode()
        self.state = self.make_current_state(self.count)
        self.get_price()
        self.player.user_action()
        self.player.render()

    def render(self):
        print(f' State:{self.state}')
        


class MarketLive():
    def __init__(self):
        pass



test = MarketSim()
test.reset()