#import sys, os

#sys.path.append('..')  
import numpy as np
from ..common import DataGrabber
from .test import Tester
from .test import Moose
from .market_live import MarketLive

#from utilities import DataGrabber
#import torch
#import numpy
class Player():
    def __init__(self, config):
        self.config = config
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
        self.spread = self.config.spread
        self.pair = self.config.pair
        self.half_spread = self.config.half_spread
        self.diff = 0
        self.live_market = MarketLive()
        self.live = False
        self.augmented = True
        #self.actions = [self.open_position_long(self.m_price), self.open_position_short(self.m_price), self.close_position(self.m_price), self.hold_position(self.m_price)]
        


    def update(self, m_price, pm_price):
        #bm_price = self.m_price
        self.m_price = m_price
        self.p_price = pm_price
        #am_price = self.m_price 
        #difference = am_price - bm_price
        #self.diff = difference * 10000
        #print(self.diff)
        self.update_placement(m_price)
        self.update_net_balance(m_price)
        self.pips = self.balance * 10000
        self.pips_net = self.net_balance * 10000
        #self.reward = 0

    def get_sl(self, pm_price):
        p_candle = pm_price
        diff = abs(p_candle[1]- p_candle[2])
        return diff

    def get_tp(self, pm_price):
        p_candle = pm_price
        diff = abs(p_candle[1]- p_candle[2])
        return diff

    def sl_check(self, state):
        if len(self.positions) == 1:
            if self.positions[0][1] == 1:
                if state[2] <= self.positions[0][2]:
                    close = self.positions[0][2] - self.half_spread
                    profit = close - self.positions[0][0]
                    self.long_positions.append([self.positions[0][0], close, profit]) 
                    self.reward = profit * 10000
                    self.balance = self.balance + profit
                    self.positions = []
                    self.placement = 0
                    return
            elif self.positions[0][1] == -1:
                if state[1] >= self.positions[0][2]:
                    close = self.positions[0][2] + self.half_spread
                    profit = self.positions[0][0] - close
                    self.short_positions.append([self.positions[0][0], close, profit])
                    self.reward = profit * 10000
                    self.balance = self.balance + profit
                    self.positions = []
                    self.placement = 0
                    return

            
            
            #self.reward = profit * 10000
            #elf.update()
        else:
            self.reward = 0
            return

    def tp_check(self, state):
        if len(self.positions) == 1:
            if self.positions[0][1] == 1:
                if state[1] >= self.positions[0][3]:
                    close = self.positions[0][3] - self.half_spread
                    profit = close - self.positions[0][0]
                    self.long_positions.append([self.positions[0][0], close, profit]) 
                    self.reward = profit * 10000
                    self.balance = self.balance + profit
                    self.positions = []
                    self.placement = 0
                    return
            elif self.positions[0][1] == -1:
                if state[2] <= self.positions[0][3]:
                    close = self.positions[0][3] + self.half_spread
                    profit = self.positions[0][0] - close
                    self.short_positions.append([self.positions[0][0], close, profit])
                    self.reward = profit * 10000
                    self.balance = self.balance + profit
                    self.positions = []
                    self.placement = 0
                    return

            
            
            #self.reward = profit * 10000
            #elf.update()
        else:
            self.reward = 0
            return



       

    def open_position_long(self, m_price, pm_price):
        if len(self.positions) == 0:
            buy = m_price + self.half_spread
            diff_sl = self.get_sl(pm_price)
            diff_tp = self.get_tp(pm_price)
            sl = m_price - diff_sl
            tp = m_price + diff_tp
            self.positions.append([(buy), 1, sl, tp])
            #self.pips += -2
            if self.live:
                    self.live_market.market_order_long()
            self.reward = 0
        if self.augmented:    
            if len(self.positions) == 1:
                self.update(m_price, pm_price)
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
                    if self.live:
                        self.live_market.position_close_short()
                        self.live_market.market_order_long()
                
            else:
                self.reward = 0
        else:
            self.reward = 0
        
    
    def open_position_short(self, m_price, pm_price):
        if len(self.positions) == 0:
            sell = m_price - self.half_spread
            diff_sl = self.get_sl(pm_price)
            diff_tp = self.get_tp(pm_price)
            sl = m_price + diff_sl
            tp = m_price - diff_tp
            self.positions.append([(sell), -1, sl, tp])
            #self.pips += -2
            if self.live:
                    self.live_market.market_order_short()
            self.reward = 0
        if self.augmented:    
            if len(self.positions) == 1:
                self.update(m_price, pm_price)
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
                    if self.live:
                        self.live_market.position_close_long()
                        self.live_market.market_order_short()
            else:
                self.reward = 0

        else:
            self.reward = 0
    
    def open_position_long_hedge(self, m_price, pm_price):
        if len(self.long_positions) == 0:
            buy = m_price + self.half_spread
            self.long_positions.append([(buy), 1])
            #self.pips += -2
            self.reward = 0
        
            
        
        
    
    def open_position_short_hedge(self, m_price, pm_price):
        if len(self.short_positions) == 0:
            sell = m_price - self.half_spread
            self.short_positions.append([(sell), -1])
            #self.pips += -2
            self.reward = 0
        
        
            
    
    def close_position(self, m_price, pm_price):
        #print(len(self.positions))
        self.update(m_price, pm_price)
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                close = m_price - self.half_spread
                profit = close - p_price
                self.long_positions.append([p_price, close, profit]) 
                if self.live:
                    self.live_market.position_close_long()
            if pos[1] == -1:
                close = m_price + self.half_spread
                profit = p_price - close
                self.short_positions.append([p_price, close, profit])
                if self.live:
                    self.live_market.position_close_short()
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
    
    def close_position_long(self, m_price, pm_price):
        #print(len(self.positions))
        self.update(m_price, pm_price)
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
        
    def close_position_short(self, m_price, pm_price):
        #print(len(self.positions))
        self.update(m_price, pm_price)
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

    
    def hold_position(self, m_price, pm_price):
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

    def action_user(self, m_price, pm_price):
        #print(len)
        #self.update(m_price)
        x = input('buy, sell, close, hold?:')
        x = str(x)
        if x == "buy":
            self.open_position_long(m_price, pm_price)
        elif  x == "sell":
            self.open_position_short(m_price, pm_price)
        elif x == "close":
            self.close_position(m_price, pm_price)
        elif x == "hold":
            self.hold_position(m_price, pm_price)
        else:
            self.hold_position(m_price, pm_price)

    def action(self, m_price, action, pm_price):
        #print(len)
        #self.update(m_price)
        x = action
        x = int(x)
        if self.placement < -0.200 or self.placement > 0.200:
            self.close_position(m_price, pm_price)
        elif x == 0:
            self.open_position_long(m_price, pm_price)
        elif x == 1:
            self.open_position_short(m_price, pm_price)
        elif x == 2:
            self.close_position(m_price, pm_price)
        elif x == 3:
            self.hold_position(m_price, pm_price)
        else:
            self.hold_position(m_price, pm_price)
    
    def action_hedge(self, m_price, action, pm_price):
        #print(len)
        #self.update(m_price)
        x = action
        x = int(x)
        if x == 0:
            self.open_position_long_hedge(m_price, pm_price)
        elif x == 1:
            self.open_position_short_hedge(m_price, pm_price)
        elif x == 2:
            self.close_position_long(m_price, pm_price)
        elif x == 3:
            self.close_position_short(m_price, pm_price)
        elif x == 4:
            self.hold_position(m_price, pm_price)
        else:
            self.hold_position(m_price, pm_price)


    def details(self, m_price):
        #self.update(m_price)
        if len(self.positions) == 1:
            pos = self.positions[0]
            if pos[1] == 1:
                return [self.balance, self.net_balance, self.placement, self.positions[0][0], self.pips_net, self.pips, [0,0,1]]
            if pos[1] == -1:
                return [self.balance, self.net_balance, self.placement, self.positions[0][0], self.pips_net, self.pips, [0,1,0]]
        else:
            return [self.balance, self.net_balance, self.placement, 0, self.pips_net, self.pips, [1,0,0]]
        
        
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
    
    
    
    
    def result(self):
        self.results = []
        self.wins = []
        self.losses = []
        self.trades = len(self.long_positions) + len(self.short_positions)
        for position in self.long_positions:
            if position[2] >= 0:
                self.wins.append(position[2])
            else:
                self.losses.append(position[2])
        for position in self.short_positions:
            if position[2] >= 0:
                self.wins.append(position[2])
            else:
                self.losses.append(position[2])
        if self.trades > 0:
            self.win_percentage = len(self.wins) / self.trades
            self.loss_percentage = len(self.losses) / self.trades
        else:
            self.win_percentage = 0
            self.loss_percentage = 0
        if len(self.wins) > 0:
            self.win_mean = np.mean(self.wins)
            self.win_median = np.median(self.wins)
            self.win_max = np.max(self.wins)
            self.win_low = np.min(self.wins)
        else:
            self.win_mean = 0
            self.win_median = 0
            self.win_max = 0
            self.win_low = 0
        if len(self.losses) > 0:
            self.loss_mean = np.mean(self.losses)
            self.loss_median = np.median(self.losses)
            self.loss_max = np.max(self.losses)
            self.loss_low = np.min(self.losses)
        else:
            self.loss_mean = 0
            self.loss_median = 0
            self.loss_max = 0
            self.loss_low = 0
        print(f'Trades: {self.trades}, wins: {len(self.wins)}, losses: {len(self.losses)}, win/loss: {self.win_percentage}/{self.loss_percentage} ')
        print(f'Win_mean: {self.win_mean}, win_median: {self.win_median}, win_max: {self.win_max}, win_low: {self.win_low}')
        print(f'loss_mean: {self.loss_mean}, loss_median: {self.loss_median}, loss_max: {self.loss_max}, loss_low: {self.loss_low}')
        #self.results.append([self.trades, len(self.wins), len(self.losses), self.win_percentage, self.loss_percentage, self.win_mean, self.loss_mean, self.win_median, self.loss_median, self.win_max, self.loss_max, self.win_low, self.loss_low])


        return self.results
    




        
    



