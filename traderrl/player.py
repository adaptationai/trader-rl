import numpy as numpy
from utilities import DataGrabber
import torch

class Player():
    def __init__(self):
        self.balance = 1000
        self.net_balance = 1000
        self.placement = 0
        self.positions = []
        self.actions = [self.open_position_long(self.m_price), self.open_position_short(self.m_price), self.close_position(self.m_price), self.hold_position(self.m_price)]
        self.m_price = None


    def update(self, m_price):
        self.m_price = m_price
        self.update_net_balance(self.m_price)
        self.update_placement(self.m_price)

    def open_position_long(self, m_price):
        if len(self.positions) == 0:
            self.positions.append([m_price, 1])
            self.net_balance += -2
    
    def open_position_short(self, m_price):
        if len(self.positions) == 0:
            self.positions.append([m_price, -1])
            self.net_balance += -2
    
    def close_position(self, m_price):
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0][0]
            if pos[0][1] == 1:
                profit = p_price - m_price 
            else:
                profit = p_price + m_price

            self.balance = self.balance + profit
            self.positions = []
    
    def hold_position(self):
        pass

    def update_placement(self, m_price):
        if len(self.positions) == 1:
            pos = self.positions[0]
            p_price = pos[0][0]
            if pos[0][1] == 1:
                profit = p_price - m_price 
            else:
                profit = p_price + m_price

            self.placement = profit
        

    def update_net_balance(self, m_price):
        pos = self.positions[0]
        price = pos[0][0]
        if pos[0][1] == 1:
            profit = price - m_price 
        else:
            profit = price + m_price

        self.net_balance = self.net_balance + profit

    def action(self, m_price):
        self.update()
        x = input('buy, sell, close, hold?:')
        if x == "buy":
            self.actions[0]
        if x == "sell":
            self.actions[1]
        if x == "buy":
            self.actions[2]
        if x == "hold":
            self.actions[3]

    def details(self):
        self.update()
        return [self.balance, self.net_balance, self.placement, self.self.postions[0]]

    def render(self):
        print(f'Player Details')
        print(f'Balance: {self.balance}')
        print(f'Net_balance: {self.net_balance}')
        print(f'Placement:: {self.placement}')
        print(f'Positions: {self.postions[0]}')




        
    



