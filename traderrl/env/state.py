import numpy as np
#from utilities import DataGrabber
import torch
from .player import Player
from ..common import DataGrabber
import numpy
import random
import json
import math
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import MarketOrderRequest, LimitOrderRequest, MITOrderRequest, PositionCloseRequest
import time
from .market_live import MarketLive
from .pairs import PairDefault 
from ..common import Auth
from ..common import Indicators
#from ..common import DataGrabber
class MarketSim():
    def __init__(self, start, eval=False, config=PairDefault()):
        self.config = config
        self.love = 14
        self.player = Player(self.config)
        self.eval = False
        self.data_grabber = DataGrabber()
        self.market_live = MarketLive()
        self.indicators = Indicators()
        self.state = None
        self.state_full = None
        self.state_current = None
        self.price = None
        self.count = 0
        self.diff = 0
        self.load = self.config.load
        self.live = self.config.live
        self.start = self.config.start
        self.end = self.config.end
        self.steps = self.config.steps
        self.pips = self.player.pips
        self.total_pips = []
        self.starts = start
        self.pm_price = 0
        
        
        self.day = True
    
    def make_episode(self):
        if self.load == True:
            self.state_full = self.data_grabber.load_state(self.starts)
        else:
            self.state_full = self.data_grabber.process_to_array()


    def make_current_state(self, count):
        if self.live == True:
            self.state = self.market_live.candles_live()
        else:
            start = (self.start+self.starter+count)
            end = (self.end+self.starter+count)
            self.state = self.state_full[start:end]
        return self.state

    def get_price(self):
        self.state_current = self.state[-1:]
        self.price = self.state_current[0][0]

    def get_pm_price(self):
        self.state_current = self.state[-2:]
        self.pm_price = self.state_current[0]
    
    def get_diff(self):
        self.state_current = self.state[-1:]
        self.diff = self.state_current[0][1]

    def step(self, action):
        #print(action)
        self.count += 1
        #self.render()
        self.get_price()
        self.get_pm_price()
        self.get_diff()
        d1 = self.player.net_balance
        pl1 = d1
        self.pips = self.player.pips
        #action = self.data_grabber.normalize(action)
        #self.player.action(self.price, action, self.pm_price)
        self.player.action(self.price, action, self.pm_price)
        if self.live:
            self.market_live.live_step_delay()
        if self.count == self.steps:
            self.player.close_position(self.price, self.pm_price, action)
        self.make_current_state(self.count)
        self.get_price()
        self.get_pm_price()
        self.get_diff()
        #self.player.sl_check(self.state[-1])
        #self.player.tp_check(self.state[-1])
        
        self.player.update(self.price, self.pm_price)
        d2 = self.player.net_balance
        pl2 = d2
        dy = d2 - d1
        rl = float(pl2) / float(pl1)
        #rl + 0.0000001
        rr= math.log(rl)
        self.reward = rr
        #self.reward = self.rewards()
        
            #self.reward = self.reward
        state = self.state_maker()
        done = self.done(self.count)
        #if self.player.balance < 1000:
            #done = True
            #self.reward = self.reward * 10
            #self.render()
        if done:
            self.pips = self.player.pips
        else:
            self.pips = 0
        
        return state, self.reward, done, self.pips


    def reset(self):
        #self.starter = 0
        self.starter = np.random.random_integers(0,96)
        self.count = 0
        self.make_episode()
        self.state = self.make_current_state(self.count)
        self.get_price()
        self.get_pm_price()
        self.get_diff()
        self.player.update(self.price, self.pm_price)
        state = self.state_maker()
        return state

    def render(self):
        #print(f' State:{self.state}')
        #print(f'Price:{self.price}')
        #print(f'diff:{self.diff}')
        #print(f'Count:{self.count}')
        #print(f'Reward:{self.reward}')
        self.player.render()
        self.player.result()
        return


    def state_maker(self):
        user = self.player.details(self.price)
        market = self.state_over_time(self.state)
        count = np.array([self.count])
        state = self.data_grabber.flatten(market, user)
        #state = self.data_grabber.flatten2(market)
        return state

    def rewards(self):
        self.delay_modifier = (self.count / self.steps)
  
        reward = self.player.balance * self.delay_modifier
        #done = self.net_worth <= 0
        return reward
    
    def done(self, count):
        if count == self.steps:
            self.render()
            return True
        elif self.player.net_balance <= 0:
            return True

        else:
            return False 

    def difference(self, state):
        new_state = []
        r = 1440
        for i in range(1440):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
        
            new_state.append([after, diff])
        return new_state

    

    def difference2(self, state):
        #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume']])
        new_state = []
        
        for i in range(24):
            c = state[i][0]
            h = state[i][1]
            l = state[i][2]
            o = state[i][3]
            v = state[i][4]
            day = state[i][5]
            hour = state[i][6]
            minute = state[i][7]
            c = c - o
            h = h - o
            l = l - o
            

            new_state.append([c, h, l, v, day, hour, minute])
        return new_state

    

    def difference3(self, state):
        #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume']])
        new_state = []
        close = state[-1][0]
        for i in range(len(state)):
    
            h = state[i][1]
            l = state[i][2]
            o = state[i][3]
            
            c = close - o 
            h = close - h
            l = close - l
        #v = state[i][4]
        #day = state[i][5]
        #hour = state[i][6]
        #minute = state[i][7]
    
            

            new_state.append([c, h, l])
        return new_state
    def state_over_time(self, state):
        #new_state2 = self.config.state_over_time(state)
        new_state = self.difference3(state)
        #so = self.indicators.stocastic_oscillator_fixed(state)
        #new_state.append([so, state[-1][0],state[-1][1], state[-1][2], state[-1][3], state[-1][4], state[-1][5], state[-1][6], state[-2][0],state[-2][1], state[-2][2], state[-2][3], state[-2][4], state[-2][5], state[-2][6], state[-3][0],state[-3][1], state[-3][2], state[-3][3], state[-3][4], state[-3][5], state[-3][6], state[-4][0],state[-4][1], state[-4][2], state[-4][3], state[-4][4], state[-4][5], state[-4][6], state[-5][0],state[-5][1], state[-5][2], state[-5][3], state[-5][4], state[-5][5], state[-5][6], state[-6][0],state[-6][1], state[-6][2], state[-6][3], state[-6][4], state[-6][5], state[-6][6], state[-7][0],state[-7][1], state[-7][2], state[-7][3], state[-7][4], state[-7][5], state[-7][6], state[-8][0],state[-8][1], state[-8][2], state[-8][3], state[-8][4], state[-8][5], state[-8][6], state[-9][0],state[-9][1], state[-9][2], state[-9][3], state[-9][4], state[-9][5], state[-9][6] ,state[-10][0],state[-10][1], state[-10][2], state[-10][3], state[-10][4], state[-10][5], state[-10][6] ,state[-11][0],state[-11][1], state[-11][2], state[-11][3], state[-11][4], state[-11][5], state[-11][6], state[-12][0], state[-12][1], state[-12][2], state[-12][3], state[-12][4], state[-12][5], state[-12][6], state[-13][0], state[-13][1], state[-13][2], state[-13][3], state[-13][4], state[-13][5], state[-13][6], state[-14][0], state[-14][1], state[-14][2], state[-14][3], state[-14][4], state[-14][5], state[-14][6]])
        return new_state

    

    


#test = MarketSim()
#test.reset()
#print(test.state[0])
#k = test.state_maker()
#print(len(k))
#for step in range(len(test.state)):
    #test.step(1)
#test = MarketLive
#test.candles_live
