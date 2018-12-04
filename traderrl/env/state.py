import numpy as np
#from utilities import DataGrabber
import torch
from .player import Player
from ..common import DataGrabber
import numpy
import random
#from ..common import DataGrabber
class MarketSim():
    def __init__(self):
        self.love = 14
        self.player = Player()
        self.actions = [1,2,3,4]
        #self.count = [1440, 2880]
        self.data_grabber = DataGrabber()
        self.state = None
        self.state_full = None
        self.state_current = None
        self.price = None
        self.count = 0
        self.diff = 0
        self.load = True
    
    def make_episode(self):
        if self.load == True:
            self.state_full = self.data_grabber.load_state_2()
        else:
            self.state_full = self.data_grabber.process_to_array()


    def make_current_state(self, count):
        start = (0+count)
        end = (60+count)
        self.state = self.state_full[start:end]
        return self.state

    def get_price(self):
        self.state_current = self.state[-1:]
        self.price = self.state_current[0][0]
    
    def get_diff(self):
        self.state_current = self.state[-1:]
        self.diff = self.state_current[0][1]

    def step(self, action):
        #self.state = self.make_current_state(self.count)
        
        #print(self.price)
        #print(self.diff)
        #self.get_price()
        self.count += 1
        #print(self.count)
        #self.render()
        #self.player.render()
    
        #print('-')
        #d1 = self.player.pips_net
        #self.player.action(self.price, action)
        self.get_price()
        self.get_diff()
        d1 = self.player.pips_net
        self.player.action(self.price, action)
        #print(self.price)
        #self.player.action_user(self.price)
        #print(self.price)
        #self.get_price()
        #print(self.price)
        #self.player.update(self.price)
        
        if self.count == 60:
            self.player.close_position(self.price)
        #self.reward = self.player.reward
        #print(self.reward)
        self.make_current_state(self.count)
        #state_diff = self.difference(self.state)
        #self.state = state_diff
        self.get_price()
        self.get_diff()
        #print(self.price)
        self.player.update(self.price)
        d2 = self.player.pips_net
        dy = d2 - d1
        #print(dy)
        #math.log(dy)
        self.reward = dy
        if self.count == 60:
            print(self.reward)
        state = self.state_maker()
        
        
        #self.get_diff()
        
        #state = self.data_grabber.scaled(state)
        
        #reward = int(reward)
        #if reward != 0:
            #self.render()
            #self.player.render()
        done = self.done(self.count)
        
        return state, self.reward, done


    def reset(self):
        self.count = 0
        self.make_episode()
        self.state = self.make_current_state(self.count)
        #self.get_price()
        #self.get_diff()
        #print(self.price)
        #self.player.update(self.price)
        #state_diff = self.difference(self.state)
        #self.state = state_diff
        self.get_price()
        self.get_diff()
        #print(self.price)
        self.player.update(self.price)
        state = self.state_maker()
        
        #print(self.price)
        #state = self.data_grabber.scaled(state)
        return state

    def render(self):
        #print(f' State:{self.state}')
        print(f'Price:{self.price}')
        #print(f'diff:{self.diff}')
        print(f'Count:{self.count}')
        print(f'Reward:{self.reward}')


    def state_maker(self):
        user = self.player.details(self.price)
        market = self.state
        #market = np.array([self.price])
        count = np.array([self.count])
        state = self.data_grabber.flatten(user, market, count)
        #state = self.data_grabber.scaled(state)
        return state

    def reward(self):

        return self.player.reward
    
    def done(self, count):
        if count == 60:
            self.render()
            self.player.render()
            return True
        else:
            return False 

    def difference(self, state):
        new_state = []
        r = 1439
        for i in range(1439):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
        
            new_state.append([after, diff])
        return new_state


    
        


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
