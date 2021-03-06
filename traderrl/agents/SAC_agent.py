import time
#moose
import gym
import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial

from stable_baselines.common.policies import MlpLnLstmPolicy, LstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv,VecNormalize 
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2, SAC
from stable_baselines.deepq import DQN
#from stable_baselines.deepq.policies import FeedForwardPolicy
from ..env import Template_Gym
from ..common import CustomPolicy, CustomPolicy_2
env = Template_Gym()
from stable_baselines.sac.policies import MlpPolicy, FeedForwardPolicy

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')
class CustomPolicy_sac(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_sac, self).__init__(*args, **kwargs,
                                           layers=[256, 256, 256],
                                            feature_extraction="mlp")
class SAC_SB():
    def __init__(self):
        self.love = 'Ramona'
        self.env_fns = [] 
        self.env_names = []
    
    def make_env(self, env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.
    
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = Template_Gym()
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    

    def train(self, num_e=1, n_timesteps=10000000, save_fraction=0.1, save='saves/m1'):
        env_id = "default"
        num_e = 32  # Number of processes to use
        # Create the vectorized environment
        #env = DummyVecEnv([lambda: env])
        #Ramona
        #self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_e)])
        env = Template_Gym()
        self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        #self.model = PPO2(CustomPolicy_2, self.env, verbose=0, learning_rate=1e-5, tensorboard_log="./test6" )
        
        self.model = SAC(CustomPolicy_sac, self.env, verbose=1, learning_rate=1e-5, tensorboard_log="./m1lstm1")
        #self.model = PPO2.load("default9", self.env, policy=CustomPolicy, tensorboard_log="./test/" )
        n_timesteps = n_timesteps * save_fraction
        n_timesteps = int(n_timesteps)
        training_loop = 1 / save_fraction
        training_loop = int(training_loop)
        
        for i in range(training_loop):
            self.model.learn(n_timesteps)
            self.model.save(save+str(i))
    
    
    def evaluate(self, num_env=32, num_steps=50, load="saves/defaultlstmday", runs=10):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        env_id = 'default'
        num_e = 1
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_env)])
        #self.model = PPO2(CustomPolicy, self.env, verbose=1, learning_rate=1e-5, tensorboard_log="./default" )
        self.env = VecNormalize(self.env, norm_obs=False, norm_reward=True)
        for i in range(runs):
            self.model = PPO2.load(load+str(i), self.env, policy=CustomPolicy_2, tensorboard_log="./default/" )
            episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
            #self.total_pips = []
            obs = self.env.reset()
            for i in range(num_steps):
                # _states are only useful when using LSTM policies
                actions, _states = self.model.predict(obs)
                # # here, action, rewards and dones are arrays
                 # # because we are using vectorized env
                obs, rewards, dones, info = self.env.step(actions)
                #self.total_pips.append(self.env.player.placement)
      
        # Stats
                for i in range(self.env.num_envs):
                    episode_rewards[i][-1] += rewards[i]
                    if dones[i]:
                        episode_rewards[i].append(0.0)

            mean_rewards =  [0.0 for _ in range(self.env.num_envs)]
            n_episodes = 0
            for i in range(self.env.num_envs):
                mean_rewards[i] = np.mean(episode_rewards[i])     
                n_episodes += len(episode_rewards[i])   

        # Compute mean reward
            mean_reward = np.mean(mean_rewards)
            print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

        return mean_reward