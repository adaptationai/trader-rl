#import argparse

import gym
import numpy as np
from traderrl import PPO2_SB
from traderrl import SAC_SB
from traderrl import DataGrabber
from traderrl import MarketLive
from traderrl import Optimization


def main():
    test = DataGrabber()
    #test.process_to_array_2()
    test.process_to_array_full()
    #test = SAC_SB()
    #test = Optimization()
    #test.optimize()
    #test = PPO2_SB()
    #test.train()
    #test.evaluate()
    #test = MarketLive()
    #test.market_order_long()
    #test.candles_live()
    #test.gen_pre_train()
    #test.pre_train()
    
    
    

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    



#ref: inc04138877 
