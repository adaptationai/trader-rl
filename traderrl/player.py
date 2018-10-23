import numpy as numpy
from utilities import DataGrabber
import torch

class Player():
    def __init__(self):
        self.balance = 0
        self.net = 0
        self.positions = []
        self.actions = []


