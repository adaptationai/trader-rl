import numpy as np


class Indicators():
    def __init__(self):
        self.ramona = 'Love'

    def fibonacci(self, candle, price):
        price_max = candle[1]
        price_min = candle[2]
        diff = price_max - price_min
        levels = [0,0,0]
        level1 = price_max - 0.236 * diff
        level2 = price_max - 0.382 * diff
        level3 = price_max - 0.681 * diff
        return diff

    
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
    
    def stocastic_oscillator_fixed(self, candles):
        #This is a basic version. Will implement better version later
        #Actually it is shit hehe
        #%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        #%D = 3-day SMA of %K
        #Lowest Low = lowest low for the look-back period
        #Highest High = highest high for the look-back period
        candles = candles[-14:]
        close = []
        high = []
        low = []
        op = []
        for i in range(len(candles)):
            close.append(candles[i][0])
            high.append(candles[i][1])
            low.append(candles[i][2])
            op.append(candles[i][3])
        c = close[-1]
        h = np.max(high) 
        l = np.min(low)
        p = (c - l) / (h - l) * 100
        return p

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
