import numpy as np
from utilities import DataGrabber
import torch
from player import Player
import numpy
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
        self.price = self.state_current[0][0]

    def step(self, action):
        #self.state = self.make_current_state(self.count)
        self.get_price()
        #self.render()
        #self.player.render()
        self.player.action(self.price, action)
        #self.player.action_user(self.price)
        self.player.update(self.price)
        self.count += 1
        self.make_current_state(self.count)
        state = self.state_maker()
        reward = self.reward()
        done = self.done(self.count)
        
        return state, reward, done


    def reset(self):
        self.count = 0
        self.make_episode()
        self.state = self.make_current_state(self.count)
        self.get_price()
        #print(self.price)
        self.player.update(self.price)
        state = self.state_maker()
        return state

    def render(self):
        #print(f' State:{self.state}')
        print(f'Price:{self.price}')
        print(f'Count:{self.count}')
        print(f'Reward:{self.player.reward}')


    def state_maker(self):
        user = self.player.details(self.price)
        market = self.state
        state = self.data_grabber.flatten(user, market)
        return state

    def reward(self):

        return self.player.reward
    
    def done(self, count):
        if count == 1439:
            return True
        else:
            return False 

    
        


class MarketLive():
    def __init__(self):
        pass



#test = MarketSim()
#test.reset()
#print(test.state[0])
#k = test.state_maker()
#print(len(k))
#for step in range(len(test.state)):
    #test.step(1)
