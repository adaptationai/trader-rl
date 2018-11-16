from state import *
from player import *
from utilities import *
from agent import Agent

class Train():
    def __init__(self):
        self.love = "love"
        self.env = MarketSim()

    def training(self):
        self.env.make_episode()
        state = self.env.make_current_state(1)
        print(self.env.state)



train = Train()
train.training()