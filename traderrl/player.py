import numpy as numpy
from utilities import DataGrabber
import torch

class Player():
    def __init__(self):
        self.balance = 1000
        self.net_balance = 1000
        self.placement = 0
        self.positions = []
        self.m_price = None
        self.difference= 0.00002
        #self.actions = [self.open_position_long(self.m_price), self.open_position_short(self.m_price), self.close_position(self.m_price), self.hold_position(self.m_price)]
        


    def update(self, m_price):
        self.m_price = m_price
        self.update_placement(self.m_price)
        self.update_net_balance(self.m_price)
        

    def open_position_long(self, m_price):
        #if len(self.positions) == 0:
        self.positions.append([(m_price - self.difference), 1])
        #self.net_balance += -0.00002
    
    def open_position_short(self, m_price):
        if len(self.positions) == 0:
            
            self.positions.append([(m_price - self.difference), -1])
            #self.net_balance += -0.00002
    
    def close_position(self, m_price):
        print(len(self.positions))
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                profit = p_price - m_price 
            if pos[1] == -1:
                profit = p_price + m_price

            self.balance = self.balance + profit
            self.positions = []
    
    def hold_position(self, m_price):
        pass

    def update_placement(self, m_price):
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0]
            if pos[1] == 1:
                profit = p_price - m_price 
            if pos[1] == -1:
                profit = p_price + m_price

            self.placement = profit
        

    def update_net_balance(self, m_price):
        #if len(self.positions) == 1:
           #pos = self.positions[0]
            #price = pos[0]
            #if pos[1] == 1:
                #profit = price - m_price 
            #else:
                #profit = price + m_price

        self.net_balance = self.balance + self.placement

    def action(self, m_price):
        #print(len)
        self.update(m_price)
        x = input('buy, sell, close, hold?:')
        x = str(x)
        if x == "buy":
            self.open_position_long(m_price)
        if x == "sell":
            self.open_position_short(m_price)
        if x == "close":
            self.close_position(m_price)
        if x == "hold":
            self.hold_position(m_price)
        else:
            self.hold_position(m_price)

    def details(self):
        self.update()
        return [self.balance, self.net_balance, self.placement, self.self.postions[0]]

    def render(self):
        print(f'Player Details')
        print(f'Balance: {self.balance}')
        print(f'Net_balance: {self.net_balance}')
        print(f'Placement:: {self.placement}')
        if len(self.positions) == 1:
            print(f'Positions: {self.positions[0]}')
        else:
            print(f'Positions: None')




        
    



