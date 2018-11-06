import numpy as np
from utilities import DataGrabber
import torch
import numpy
class Player():
    def __init__(self):
        self.balance = 0
        self.net_balance = 0
        self.placement = 0
        self.positions = []
        self.m_price = None
        self.pip = 0.0002
        self.pips = 0
        self.pips_net = 0
        self.reward = 0
        #self.actions = [self.open_position_long(self.m_price), self.open_position_short(self.m_price), self.close_position(self.m_price), self.hold_position(self.m_price)]
        


    def update(self, m_price):
        self.m_price = m_price
        self.update_placement(self.m_price)
        self.update_net_balance(self.m_price)
        self.pips = self.balance * 10000
        self.pips_net = self.net_balance * 10000
        #self.reward = 0
        

    def open_position_long(self, m_price):
        if len(self.positions) == 0:
            self.positions.append([(m_price - self.pip), 1])
            self.reward = 0
        
    
    def open_position_short(self, m_price):
        if len(self.positions) == 0:
            self.positions.append([(m_price - self.pip), -1])
            self.reward = 0
    
    def close_position(self, m_price):
        print(len(self.positions))
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                profit = m_price - p_price 
            if pos[1] == -1:
                profit = p_price - m_price

            self.balance = self.balance + profit
            self.positions = []
            self.placement = 0
            self.reward = profit * 10000
    
    def hold_position(self, m_price):
        #pass
        self.reward = 0

    def update_placement(self, m_price):
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                profit = m_price - p_price
            if pos[1] == -1:
                profit = p_price - m_price

            self.placement = profit
        

    def update_net_balance(self, m_price):
        self.net_balance = self.balance + self.placement

    def action_user(self, m_price):
        #print(len)
        self.update(m_price)
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
        self.update(m_price)
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

    def details(self, m_price):
        self.update(m_price)
        if len(self.positions) == 1:
            return [self.balance, self.net_balance, self.placement, self.self.postions[0]]
        else:
            return [self.balance, self.net_balance, self.placement, [0,0]]
        

    def render(self):
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




        
    



