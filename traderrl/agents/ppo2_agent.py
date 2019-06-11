import time

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
from ..common import CustomPolicy, CustomPolicy_2, CustomLSTMPolicy, CustomPolicy_4
env = Template_Gym()
from stable_baselines.gail import generate_expert_traj

from stable_baselines.gail import ExpertDataset


timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

class PPO2_SB():
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
    

    def train(self, num_e=1, n_timesteps=1000000, save_fraction=0.1, save='saves/aud5'):
        env_id = "default"
        num_e = 1  # Number of processes to use
        # Create the vectorized environment
        #env = DummyVecEnv([lambda: env])
        #Ramona
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_e)])
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        self.model = PPO2(CustomPolicy_4, self.env, verbose=0, learning_rate=1e-4, nminibatches=1, tensorboard_log="./day1" )
        
        
        #self.model = PPO2.load("default9", self.env, policy=CustomPolicy, tensorboard_log="./test/" )
        n_timesteps = n_timesteps * save_fraction
        n_timesteps = int(n_timesteps)
        training_loop = 1 / save_fraction
        training_loop = int(training_loop)
        log_dir = "saves"
        for i in range(training_loop):
            self.model.learn(n_timesteps)
            self.model.save(save+str(i))
            self.env.save_running_average(log_dir)
        self.env.save_running_average(log_dir)
    
    
    def evaluate(self, num_env=1, num_steps=21900, load="saves/aud5", runs=10):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        env_id = 'default'
        num_e = 1
        log_dir = "saves"
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_env)])
        #self.model = PPO2(CustomPolicy, self.env, verbose=1, learning_rate=1e-5, tensorboard_log="./default" )
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        self.env.load_running_average(log_dir)
        for i in range(runs):
            self.model = PPO2.load(load+str(i), self.env, policy=CustomPolicy_4,  tensorboard_log="./default/" )
            self.env.load_running_average(log_dir)
            episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
            #self.total_pips = []
            obs = self.env.reset()
            state = None
            # When using VecEnv, done is a vector
            done = [False for _ in range(env.num_envs)]
            for i in range(num_steps):
                # _states are only useful when using LSTM policies
                action, state = self.model.predict(obs, state=state, mask=done, deterministic=True)
                obs, rewards , dones, _ = self.env.step(action)
                #actions, _states = self.model.predict(obs)
                # # here, action, rewards and dones are arrays
                 # # because we are using vectorized env
                #obs, rewards, dones, info = self.env.step(actions)
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

    def pre_train(self, num_e=1, load="saves/m19"):
        env_id = 'default'
        num_e = 1
        log_dir = "saves"
        # Usingenv = make_env() only one expert trajectory
        # you can specify `traj_limitation=-1` for using the whole dataset
        dataset = ExpertDataset(expert_path='default2.npz',traj_limitation=1, batch_size=128)
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_e)])
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        #env = make_env()
        #model = GAIL("MlpPolicy", env=env, expert_dataset=dataset, verbose=1)
        self.env.load_running_average("saves")
        self.model = PPO2(CustomPolicy, self.env, verbose=1, nminibatches=1,  learning_rate=1e-5, tensorboard_log="./m1ln4" )
        #self.model = PPO2.load("saves/m19", self.env, policy=CustomPolicy, tensorboard_log="./default/" )
        self.env.load_running_average("saves")
        # Pretrain the PPO2 model
        self.model.pretrain(dataset, n_epochs=10000)

        # As an option, you can train the RL agent
        #self.model.learn(int(100000000))

        # Test the pre-trained model
        self.env = self.model.get_env()
        self.env.load_running_average("saves")
        obs = self.env.reset()

        reward_sum = 0.0
        for _ in range(1000000):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.env.step(action)
            reward_sum += reward
            #self.env.render()
            if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = self.env.reset()

        self.env.close()

    def gen_pre_train(self, num_e=1, save='default2', episodes=1000):
        #self.create_envs(game_name=game, state_name=state, num_env=num_e)
        #self.env=SubprocVecEnv(self.env_fns)
        env_id = 'default'
        num_e = 1
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_e)])
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        #env = make_env()
        #model = GAIL("MlpPolicy", env=env, expert_dataset=dataset, verbose=1)
        self.env.load_running_average("saves")
        self.model = PPO2.load("saves/m19", self.env, policy=CustomPolicy, tensorboard_log="./default/" )
        self.env.load_running_average("saves")
        #env = make_env()
        #self.expert_agent = 
        generate_expert_traj(self.model, save, self.env, n_episodes=episodes)
        

