import numpy as np
#from utilities import DataGrabber
import torch
from .player import Player
from ..common import DataGrabber
import numpy
import random
import json
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import MarketOrderRequest, LimitOrderRequest, MITOrderRequest, PositionCloseRequest
import time

from .auth import Auth
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
            self.state_full = self.data_grabber.load_state()
        else:
            self.state_full = self.data_grabber.process_to_array()


    def make_current_state(self, count):
        start = (0+count)
        end = (1440+count)
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
        self.count += 5
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
        
        if self.count == 1440:
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
        #print(f'Price:{self.price}')
        #print(f'diff:{self.diff}')
        #print(f'Count:{self.count}')
        #print(f'Reward:{self.reward}')
        pass


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
        if count == 1440:
            self.render()
            self.player.render()
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
        cl5 = state[-5][0]
        hi5 = state[-5][1]
        lo5 = state[-5][2]
        op5 =state[-5][3]
        v5 = state[-5][4]
        day5 = state[-5][5]
        hour5 = state[-5][6]
        minute5 = state[-5][7]
        cl6 = state[-6][0]
        hi6 = state[-6][1]
        lo6 = state[-6][2]
        op6 =state[-6][3]
        v6 = state[-6][4]
        day6 = state[-6][5]
        hour6 = state[-6][6]
        minute6 = state[-6][7]
        cl7 = state[-7][0]
        hi7 = state[-7][1]
        lo7 = state[-7][2]
        op7 =state[-7][3]
        v7 = state[-7][4]
        day7 = state[-7][5]
        hour7 = state[-7][6]
        minute7 = state[-7][7]
        cl8 = state[-8][0]
        hi8 = state[-8][1]
        lo8 = state[-8][2]
        op8 =state[-8][3]
        v8 = state[-8][4]
        day8 = state[-8][5]
        hour8 = state[-8][6]
        minute8 = state[-8][7]
        cl9 = state[-9][0]
        hi9 = state[-9][1]
        lo9 = state[-9][2]
        op9 =state[-9][3]
        v9 = state[-9][4]
        day9 = state[-9][5]
        hour9 = state[-9][6]
        minute9 = state[-9][7]
        cl10 = state[-10][0]
        hi10 = state[-10][1]
        lo10 = state[-10][2]
        op10 =state[-10][3]
        v10 = state[-10][4]
        day10 = state[-10][5]
        hour10 = state[-10][6]
        minute10 = state[-10][7]
        #cl5 = state[-5][0]
        #cl15 = state[-15][0]
        cl5 =  cl - state[-5][3]
        cl15 =  cl - state[-15][3]
        cl30 =  cl - state[-30][3]
        cl1h = cl - state[-60][3]
        cl2h = cl - state[-120][3]
        cl4h = cl - state[-240][3]
        cl8h = cl - state[-480][3]
        cl16h = cl - state[-960][3]
        clday = cl - state[-1440][3]
        clnow = cl - op
        hinow = hi - op
        lonow = lo - op

        state5 = state[-5:]
        state15 = state[-15:]
        state30 = state[-30:]
        state1h = state[-60:]
        state2h = state[-120:]
        state4h = state[-240:]
        state8h = state[-480:]
        state16h = state[-960:]
        stateday = state[-1440:]
        statediff = self.difference2(state)
        state5diff = self.difference2(state5)
        state15diff = self.difference2(state15)
        state30diff = self.difference2(state30)
        state1hdiff = self.difference2(state1h)
        state2hdiff = self.difference2(state2h)
        state4hdiff = self.difference2(state4h)
        state8hdiff = self.difference2(state8h)
        state16hdiff = self.difference2(state16h)
        statedaydiff = self.difference2(stateday)
        av = self.average_diff(statediff)
        av5 = self.average_diff(state30diff)
        av15 = self.average_diff(state30diff)
        av30 = self.average_diff(state30diff)
        av1h = self.average_diff(state1hdiff)
        av2h = self.average_diff(state2hdiff)
        av4h = self.average_diff(state4hdiff)
        av8h = self.average_diff(state8hdiff)
        av16h = self.average_diff(state16hdiff)
        avday = self.average_diff(statedaydiff)
        md5 = self.median_diff(state5diff)
        md15 = self.median_diff(state15diff)
        md30 = self.median_diff(state30diff)
        md1h = self.median_diff(state1hdiff)
        md2h = self.median_diff(state2hdiff)
        md4h = self.median_diff(state4hdiff)
        md8h = self.median_diff(state8hdiff)
        md16h = self.median_diff(state16hdiff)
        mdday = self.median_diff(statedaydiff)
        atr = self.atr(state)
        atr5 = self.atr(state5)
        atr15 = self.atr(state15)
        atr30 = self.atr(state30)
        atr1h = self.atr(state1h)
        atr2h = self.atr(state2h)
        atr4h = self.atr(state4h)
        atr8h = self.atr(state8h)
        atr16h = self.atr(state16h)
        atrday = self.atr(stateday)
        aavol = self.average_vol(state)
        aavol5 = self.average_vol(state5)
        aavol15 = self.average_vol(state15)
        aavol30 = self.average_vol(state30)
        aavol1h = self.average_vol(state1h)
        aavol2h = self.average_vol(state2h)
        aavol4h = self.average_vol(state4h)
        aavol8h = self.average_vol(state8h)
        aavol16h = self.average_vol(state16h)
        aavolday = self.average_vol(stateday)
        #candle1 = self.candle_maker(state[-1:])
        candle5 = self.candle_maker(state[-5:])
        
        candle15 = self.candle_maker(state[-15:])
        candle30 = self.candle_maker(state[-30:])
        candle1h = self.candle_maker(state[-60:])
        candle2h = self.candle_maker(state[-120:])
        candle4h = self.candle_maker(state[-240:])
        candle8h = self.candle_maker(state[-480:])
        candle16h = self.candle_maker(state[-960:])
        candleday = self.candle_maker(state[-1440:])

        #so = self.stocastic_oscillator(candle1)
        so5 = self.stocastic_oscillator(candle5)
        #print(so5)
        so15 = self.stocastic_oscillator(candle15)
        so30 = self.stocastic_oscillator(candle30)
        so1h = self.stocastic_oscillator(candle1h)
        so2h = self.stocastic_oscillator(candle2h)
        so4h = self.stocastic_oscillator(candle4h)
        so8h = self.stocastic_oscillator(candle8h)
        so16h = self.stocastic_oscillator(candle16h)
        soday = self.stocastic_oscillator(candleday)
        

        
        

        #new_state.append([cl, hi, lo, op, v, day, hour, minute, cl2, hi2, lo2, op2, v2, day2, hour2, minute2, cl3, hi3, lo3, op3, v3, day3, hour3, minute3, cl4, hi4, lo4, op4, v4, day4, hour4, minute4, cl5, hi5, lo5, op5, v5, day5, hour5, minute5, cl6, hi6, lo6, op6, v6, day6, hour6, minute6, cl7, hi7, lo7, op7, v7, day7, hour7, minute7, cl8, hi8, lo8, op8, v8, day8, hour8, minute8, cl9, hi9, lo9, op9, v9, day9, hour9, minute9, cl10, hi10, lo10, op10, v10, day10, hour10, minute10, clnow, hinow, lonow, cl30, cl1h, cl2h, cl4h, cl8h, cl16h, clday, atr14, atr30, atr1h, atr2h, atr4h, atr8h, atr16h, atrday, av30, av1h, av2h, av4h, av8h, av16h, avday, md30, md1h, md2h, md4h, md8h, md16h, mdday])
        #new_state.append([cl, hi, lo, op, v, day, hour, minute, clnow, hinow, lonow, cl30, cl1h, cl2h, cl4h, cl8h, cl16h, clday, atr15, atr30, atr1h, atr2h, atr4h, atr8h, atr16h, atrday, av30, av1h, av2h, av4h, av8h, av16h, avday, md30, md1h, md2h, md4h, md8h, md16h, mdday])
        new_state.append([cl, hi, lo, op, v, day, hour, minute, clnow, hinow, lonow, cl5, cl15, cl30, cl1h, cl2h, cl4h, cl8h, cl16h, clday, atr, atr5, atr15, atr30, atr1h, atr2h, atr4h, atr8h, atr16h, atrday, av5, av15, av30, av1h, av2h, av4h, av8h, av16h, avday, md5, md15, md30, md1h, md2h, md4h, md8h, md16h, mdday, aavol, aavol5, aavol15, aavol30, aavol1h, aavol2h, aavol4h, aavol8h, aavol16h, aavolday, so1h, so2h, so4h, so8h, so16h, soday])

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

    def stocastic_oscillator(self, candle):
        c = candle[0]
        h = candle[1]
        l = candle[2]
        o = candle[3]
        k = c - l
        m = h - l
        p =  k / m
        return p

    def candle_maker(self, state):
        new_state = []
        highs = []
        lows = []
        new_state.append(state[-1][0])
        for i in range(len(state)):
            highs.append(state[i][1])
        for i in range(len(state)):
            lows.append(state[i][2])
        new_state.append(max(highs))
        new_state.append(min(lows))
        new_state.append(state[0][3])

        

        return new_state




class MarketLive():
    def __init__(self):
        self.accountID = Auth.accountID 
        self.access_token = Auth.access_token

    def market_order(self):
        client = API(access_token=self.access_token)
        mo = MarketOrderRequest(instrument="EUR_USD", units=10000)
        r = orders.OrderCreate(self.accountID, data=mo.data)
        rv = client.request(r)

    def limit_order(self):
        client = API(access_token=self.access_token)
        ordr = LimitOrderRequest(instrument="EUR_USD", units=10000, price=1.08)
        r = orders.orderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def market_if_touched(self):
        client = API(access_token=self.access_token)
        ordr = MITOrderRequest(instrument="EUR_USD", units=10000, price=1.08)
        r = orders.OrderCreate(self.accountID, data=ordr.data)
        rv = client.request(r)

    def position_close(self):
        client = API(access_token=self.access_token)
        ordr = PositionCloseRequest(longUnits=10000)
        r = position.PositionClose(self.accountID, instrument="EUR_USD", data=ordr.data)
        rv = client.request(r)

    def candles_live(self):
        client = API(access_token=self.access_token)
        params = {"count": 4320, "granularity": "M1"}
        r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
        data = client.request(r)
        data = self.data_grabber.data_converted(data)
        data = self.data_grabber.time_to_array(data)
        data = self.data_grabber.toarray(data)
        return data

    def live_step_delay(self):
        current_time = time.time()
        time_to_sleep = 60 - (current_time % 60)
        time.sleep(time_to_sleep)


#test = MarketSim()
#test.reset()
#print(test.state[0])
#k = test.state_maker()
#print(len(k))
#for step in range(len(test.state)):
    #test.step(1)
#test = MarketLive
#test.candles_live
