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
        end = (96+count)
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
        d1 = self.player.net_balance
        pl1 = d1+10000
        self.player.action(self.price, action)
        #print(self.price)
        #self.player.action_user(self.price)
        #print(self.price)
        #self.get_price()
        #print(self.price)
        #self.player.update(self.price)
        
        if self.count == 96:
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
        d2 = self.player.net_balance
        #print(d)
        #print(d2)
        pl2 = d2+10000
        dy = d2 - d1
        #print(dy)
        rl = float(pl2) / float(pl1)
        #print(self.count)
        #print(rl)
        rr= math.log(rl)
        #print(rl)
        self.reward = rr
        #if self.count == 96:
            #print(self.reward)
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
        #market = self.state
        #market = self.difference2(self.state)
        market = self.state_over_time(self.state)
        #market = np.array([self.price])
        count = np.array([self.count])
        state = self.data_grabber.flatten(market, user, count)
        #state = self.data_grabber.scaled(state)
        return state

    def reward2(self):

        return self.player.reward
    
    def done(self, count):
        if count == 96:
            self.render()
            self.player.render()
            return True
        else:
            return False 

    def difference(self, state):
        new_state = []
        r = 96
        for i in range(96):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
        
            new_state.append([after, diff])
        return new_state

    def difference2(self, state):
        #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume']])
        new_state = []
        
        for i in range(len(state)):
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

    def state_over_time(self, state):
        new_state = []
        cl = state[-1][0]
        hi = state[-1][1]
        lo = state[-1][2]
        op =state[-1][3]
        v = state[-1][4]
        day = state[-1][5]
        hour = state[-1][6]
        minute = state[-1][7]
        cl2 = state[-2][0]
        hi2 = state[-2][1]
        lo2 = state[-2][2]
        op2 =state[-2][3]
        v2 = state[-2][4]
        day2 = state[-2][5]
        hour2 = state[-2][6]
        minute2 = state[-2][7]
        cl3 = state[-3][0]
        hi3 = state[-3][1]
        lo3 = state[-3][2]
        op3 =state[-3][3]
        v3 = state[-3][4]
        day3 = state[-3][5]
        hour3 = state[-3][6]
        minute3 = state[-3][7]
        cl4 = state[-4][0]
        hi4 = state[-4][1]
        lo4 = state[-4][2]
        op4 =state[-4][3]
        v4 = state[-4][4]
        day4 = state[-4][5]
        hour4 = state[-4][6]
        minute4 = state[-4][7]
        #cl5 = state[-5][0]
        #cl15 = state[-15][0]
        cl30 =  cl - state[-2][3]
        cl1h = cl - state[-4][3]
        cl2h = cl - state[-8][3]
        cl4h = cl - state[-16][3]
        cl8h = cl - state[-32][3]
        cl16h = cl - state[-64][3]
        clday = cl - state[-96][3]
        clnow = cl - op
        hinow = hi - op
        lonow = lo - op
        state14 = state[-14:]
        state30 = state[-2:]
        state1h = state[-4:]
        state2h = state[-8:]
        state4h = state[-16:]
        state8h = state[-32:]
        state16h = state[-64:]
        stateday = state[-96:]
        state30diff = self.difference2(state30)
        state1hdiff = self.difference2(state1h)
        state2hdiff = self.difference2(state2h)
        state4hdiff = self.difference2(state4h)
        state8hdiff = self.difference2(state8h)
        state16hdiff = self.difference2(state16h)
        statedaydiff = self.difference2(stateday)
        av30 = self.average_diff(state30diff)
        av1h = self.average_diff(state1hdiff)
        av2h = self.average_diff(state2hdiff)
        av4h = self.average_diff(state4hdiff)
        av8h = self.average_diff(state8hdiff)
        av16h = self.average_diff(state16hdiff)
        avday = self.average_diff(statedaydiff)
        md30 = self.median_diff(state30diff)
        md1h = self.median_diff(state1hdiff)
        md2h = self.median_diff(state2hdiff)
        md4h = self.median_diff(state4hdiff)
        md8h = self.median_diff(state8hdiff)
        md16h = self.median_diff(state16hdiff)
        mdday = self.median_diff(statedaydiff)
        atr14 = self.atr(state14)
        atr30 = self.atr(state30)
        atr1h = self.atr(state1h)
        atr2h = self.atr(state2h)
        atr4h = self.atr(state4h)
        atr8h = self.atr(state8h)
        atr16h = self.atr(state16h)
        atrday = self.atr(stateday)
        

        new_state.append([cl, hi, lo, op, v, day, hour, minute, cl2, hi2, lo2, op2, v2, day2, hour2, minute2, cl3, hi3, lo3, op3, v3, day3, hour3, minute3, cl4, hi4, lo4, op4, v4, day4, hour4, minute4, clnow, hinow, lonow, cl30, cl1h, cl2h, cl4h, cl8h, cl16h, clday, atr14, atr30, atr1h, atr2h, atr4h, atr8h, atr16h, atrday, av30, av1h, av2h, av4h, av8h, av16h, avday, md30, md1h, md2h, md4h, md8h, md16h, mdday])

        return new_state

    def average_diff(self, state):
    
        new_state = []
        for i in range(len(state)):
            new_state.append(state[i][0])
        new_state = np.mean(new_state)
        return new_state


    def median_diff(self, state):
        new_state = []
        for i in range(len(state)):
            new_state.append(state[i][0])
        new_state = np.median(new_state)
        return new_state
    
    def atr(self, state):
        stateatr = []
        for i in range(len(state)):
            stateatr.append(state[i][1])
            stateatr.append(state[i][2])
        highatr = max(stateatr)
        lowatr = min(stateatr)
        atr = highatr - lowatr
        return atr  
    
    def average_vol(self, state):
    
        new_state = []
        for i in range(len(state)):
            new_state.append(state[i][4])
        new_state = np.mean(new_state)
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
