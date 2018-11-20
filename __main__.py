import argparse

import gym
import numpy as np
from traderrl import DataGrabber
from traderrl import MarketSim
from traderrl import Template
from traderrl import Template_Gym
from traderrl import Baseline_Details
from traderrl import 
#test = Baseline_Details()
test from

def main():
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=10000000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()

    test.main(args)
    #test = Template_Gym()
    #test = Template()
    #test.reset()
    #print(test.state[0])
        #k = test.state_maker()
#print(len(k))
    #for step in range(len(test.state)):
        #test.step(1)
    


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
