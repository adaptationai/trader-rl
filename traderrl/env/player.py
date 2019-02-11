#import sys, os

#sys.path.append('..')  
import numpy as np
from ..common import DataGrabber
#from utilities import DataGrabber
import torch
import numpy
class Player():
    def __init__(self):
        self.balance = 0
        self.net_balance = 0
        self.placement = 0
        self.positions = []
        self.short_positions = []
        self.long_positions = []
        self.m_price = 0
        self.pip = 0.0002
        self.pips = 0
        self.pips_net = 0
        self.reward = 0
        self.spread = 0.0002
        self.half_spread = 0.0001
        self.diff = 0
        #self.actions = [self.open_position_long(self.m_price), self.open_position_short(self.m_price), self.close_position(self.m_price), self.hold_position(self.m_price)]
        


    def update(self, m_price):
        #bm_price = self.m_price
        self.m_price = m_price
        #am_price = self.m_price 
        #difference = am_price - bm_price
        #self.diff = difference * 10000
        #print(self.diff)
        self.update_placement(m_price)
        self.update_net_balance(m_price)
        self.pips = self.balance * 10000
        self.pips_net = self.net_balance * 10000
        #self.reward = 0
        

    def open_position_long(self, m_price):
        if len(self.positions) == 0:
            buy = m_price + self.half_spread
            self.positions.append([(buy), 1])
            #self.pips += -2
            self.reward = 0
        elif len(self.positions) == 1:
            self.update(m_price)
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == -1:
                close = m_price + self.half_spread
                profit = p_price - close
                #print('m_price')
                #print(m_price)
                #print('placement')
                #print(self.placement)
                self.reward = profit * 10000
                self.balance = self.balance + profit
                self.positions = []
                self.placement = 0
                buy = m_price + self.half_spread
                self.positions.append([(buy), 1])
                #self.pips += -2
                self.reward = 0
                #self.reward = profit * 10000
                #elf.update()
            else:
                self.reward = 0
        else:
            self.reward = 0
        
    
    def open_position_short(self, m_price):
        if len(self.positions) == 0:
            sell = m_price - self.half_spread
            self.positions.append([(sell), -1])
            #self.pips += -2
            self.reward = 0
        elif len(self.positions) == 1:
            self.update(m_price)
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                close = m_price - self.half_spread
                profit = close - p_price
                #print('m_price')
                #print(m_price)
                #print('placement')
                #print(self.placement)
                self.reward = profit * 10000
                self.balance = self.balance + profit
                self.positions = []
                self.placement = 0
                sell = m_price - self.half_spread
                self.positions.append([(sell), -1])
                #self.pips += -2
                self.reward = 0
                #self.reward = profit * 10000
                #elf.update()
            else:
                self.reward = 0

        else:
            self.reward = 0
    
    def open_position_long_hedge(self, m_price):
        if len(self.long_positions) == 0:
            buy = m_price + self.half_spread
            self.long_positions.append([(buy), 1])
            #self.pips += -2
            self.reward = 0
        
            
        
        
    
    def open_position_short_hedge(self, m_price):
        if len(self.short_positions) == 0:
            sell = m_price - self.half_spread
            self.short_positions.append([(sell), -1])
            #self.pips += -2
            self.reward = 0
        
        
            
    
    def close_position(self, m_price):
        #print(len(self.positions))
        self.update(m_price)
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                close = m_price - self.half_spread
                profit = close - p_price
                self.long_positions.append([price, close, profit]) 
            if pos[1] == -1:
                close = m_price + self.half_spread
                profit = p_price - close
                self.short_positions([price, close, profit])
            #print('m_price')
            #print(m_price)
            #print('placement')
            #print(self.placement)
            self.reward = profit * 10000
            self.balance = self.balance + profit
            self.positions = []
            self.placement = 0
            #self.reward = profit * 10000
            #elf.update()
        else:
            self.reward = 0
    
    def close_position_long(self, m_price):
        #print(len(self.positions))
        self.update(m_price)
        if len(self.long_positions) == 1:
            pos = self.long_positions[0]
            p_price = pos[0]
            close = m_price - self.half_spread
            profit = close - p_price 
            
            #print('m_price')
            #print(m_price)
            #print('placement')
            #print(self.placement)
            self.reward = profit * 10000
            self.balance = self.balance + profit
            self.long_positions = []
            self.update_placement_hedge(m_price)
            #self.reward = profit * 10000
            #elf.update()
        
    def close_position_short(self, m_price):
        #print(len(self.positions))
        self.update(m_price)
        if len(self.short_positions) == 1:
            pos = self.short_positions[0]
            p_price = pos[0]
            close = m_price + self.half_spread
            profit = p_price - close
            #print('m_price')
            #print(m_price)
            #print('placement')
            #print(self.placement)
            self.reward = profit * 10000
            self.balance = self.balance + profit
            self.short_positions = []
            self.update_placement_hedge(m_price)
            #self.reward = profit * 10000
            #elf.update()
        else:
            self.reward = 0

    
    def hold_position(self, m_price):
        #pass
        self.reward = 0

    def update_placement(self, m_price):
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                close = m_price - self.half_spread
                profit = close - p_price 
            if pos[1] == -1:
                close = m_price + self.half_spread
                profit = p_price - close
            
            self.placement = profit

    def update_placement_hedge(self, m_price):
        if len(self.long_positions) == 1:
            pos = self.long_positions[0]
            p_price = pos[0]
            close = m_price - self.half_spread
            long_profit = close - p_price
        else:
            long_profit = 0
        if len(self.short_positions) == 1:
            pos = self.short_positions[0]
            p_price = pos[0]
            close = m_price + self.half_spread
            short_profit = p_price - close
        else:
            short_profit = 0 
           
            
        self.placement = long_profit+short_profit
        

    def update_net_balance(self, m_price):
        self.net_balance = self.balance + self.placement

    def action_user(self, m_price):
        #print(len)
        #self.update(m_price)
        x = input('buy, sell, close, hold?:')
        x = str(x)
        if x == "buy":
            self.open_position_long(m_price)
        elif  x == "sell":
            self.open_position_short(m_price)
        elif x == "close":
            self.close_position(m_price)
        elif x == "hold":
            self.hold_position(m_price)
        else:
            self.hold_position(m_price)

    def action(self, m_price, action):
        #print(len)
        #self.update(m_price)
        x = action
        x = int(x)
        if x == 0:
            self.open_position_long(m_price)
        elif x == 1:
            self.open_position_short(m_price)
        elif x == 2:
            self.close_position(m_price)
        elif x == 3:
            self.hold_position(m_price)
        else:
            self.hold_position(m_price)
    
    def action_hedge(self, m_price, action):
        #print(len)
        #self.update(m_price)
        x = action
        x = int(x)
        if x == 0:
            self.open_position_long_hedge(m_price)
        elif x == 1:
            self.open_position_short_hedge(m_price)
        elif x == 2:
            self.close_position_long(m_price)
        elif x == 3:
            self.close_position_short(m_price)
        elif x == 4:
            self.hold_position(m_price)
        else:
            self.hold_position(m_price)

    def details(self, m_price):
        #self.update(m_price)
        if len(self.positions) == 1:
            pos = self.positions[0]
            if pos[1] == 1:
                return [self.balance, self.net_balance, self.placement, self.pips_net, [0,0,1]]
            if pos[1] == -1:
                return [self.balance, self.net_balance, self.placement, self.pips_net, [0,1,0]]
        else:
            return [self.balance, self.net_balance, self.placement, self.pips_net, [1,0,0]]
        
    def details_hedge(self, m_price):
        #self.update(m_price)
        if len(self.long_positions) == 1 and len(self.short_positions) == 1:
            
            return [self.balance, self.net_balance, self.placement, self.pips_net, [0,1,1]]
        if len(self.long_positions) == 1:
            return [self.balance, self.net_balance, self.placement, self.pips_net, [0,1,0]]

        if len(self.short_positions) == 1:
            return [self.balance, self.net_balance, self.placement, self.pips_net, [0,0,1]]
        else:
            return [self.balance, self.net_balance, self.placement, self.pips_net, [1,0,0]]

    def render(self):
        #self.update(self.m_price)
        print(f'Player Details')
        print(f'Pips: {self.pips}')
        print(f'Pips_net: {self.pips_net}')
        print(f'Balance: {self.balance}')
        print(f'Net_balance: {self.net_balance}')
        print(f'Placement:: {self.placement}')
        if len(self.positions) == 1:
            print(f'Positions: {self.positions[0]}')
        else:
            print(f'Positions: None')
    
    
    
    def results(self):
        self.results = []
        self.wins = []
        self.losses = []
        self.trades = len(self.long_positions) + len(self.short_positions)
        for position in self.long_positions:
            if position[2] >= 0:
                self.wins.append(position[2]
            else:
                self.losses.append(position[2])
        for position in self.short_positions:
            if position[2] >= 0:
                self.wins.append(position[2]
            else:
                self.loses.append(position[2]
        self.win_percentage = len(self.wins) / len(self.trades)
        self.loss_percentage = len(self.losses) / len(self.trades)
        self.win_mean = np.mean(self.wins)
        self.loss_mean = np.mean(self.losses)
        self.win_median = np.median(self.wins)
        self.loss_median = np.median(self.losses)
        self.win_max = np.max(self.wins)
        self.loss_max = np.max(self.loses)
        self.win_low = np.min(self.wins)
        self.loss_low = np.min(self.loses)
        #print(f'Trades: {self.trades}, wins: {len(self.wins)}, losses: {len(self.losses)}, win/loss: {self.win_percentage}/{self.loss_percentage} ')
        #print(f'Win_mean: {self.win_mean}, win_median: {self.win_median}, win_max: {self.win_max}, win_low: {self.win_low}')
        #print(f'loss_mean: {self.loss_mean}, loss_median: {self.loss_median}, loss_max: {self.loss_max}, loss_low: {self.loss_low}')
        self.results.append([self.trades, len(self.wins), len(self.losses), self.win_percentage, self.loss_percentage, self.win_mean, self.loss_mean, self.win_median, self.loss_median, self.win_max, self.loss_max, self.win_low, self.loss_low])


        return 
    




        
    



