import numpy as numpy
from utilities import DataGrabber
import torch

class Player():
    def __init__(self):
        self.balance = 0
        self.net = 0
        self.positions = []
        self.actions = []
        self.position = []

    def open_position_long(self):
        pass
    
    def open_position_short(self):
        pass
    
    def close_position(self, m_price):
        pos = self.positions[0]
        price = pos[0]
        profit = price - m_price 
        self.balance = self.balance + profit
    
    def hold_position(self):
        pass

    def net_balance(self, m_price):
        pos = self.positions[0]
        price = pos[0]
        net = price - m_price
        return net


