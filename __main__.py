#import argparse

import gym
import numpy as np
from traderrl import PPO2_SB
from traderrl import DataGrabber
from traderrl import MarketLive
#test = Baseline_Details()
#test = Train51()

def main():
    #test = DataGrabber()
    #test.process_to_array_2()
    test = PPO2_SB()
    #test.train()
    test.evaluate()
    #test = MarketLive()
    #test.position_close_long()
    
    
    

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    



#ref: inc04138877 
