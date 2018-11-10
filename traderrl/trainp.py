#from model import DQN
from torch.autograd import Variable
#from torchviz import make_dot
import numpy as np
from collections import deque
#import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
from collections import namedtuple, deque

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#is_ipython = 'inline' in plt.get_backend()
#if is_ipython:
#    from IPython import display
#plt.ion()

from deep_rl import *
import torch
import numpy as np
from collections import deque
#from agent import Agent
from env import Template

env = Template("trader")

def run_steps_2(agent):
    random_seed()
    config = agent.config
    save_path = '1000_update_trader_placement.bin'
    agent.load(save_path)
    while True:
        rewards = agent.rewards_deque
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):            
            config.logger.info('Mean return of %.2f over a window of %d recent episodes. %d episodes played so far, most recent episodic return is %.2f' % (
                np.mean(rewards), len(rewards), len(agent.rewards_all), agent.rewards_all[-1]))
            save_path = '1000_update_trader_placement.bin'
            agent.save(save_path)

        if len(rewards) >= 100 and agent.rewards_all[-1] % 100 == 0:
            save_path = '1000_update_trader_placement.bin'
            agent.save(save_path)
            
        
        if len(rewards) and (np.mean(rewards) >= 100):
            save_path = 'win_trader.bin'
            agent.save(save_path)
            res = True, agent.rewards_deque, agent.rewards_all, save_path
            agent.close()
            return res
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            return False, None, None, None
        agent.step()




class Trader(BaseTask):
    def __init__(self):
        BaseTask.__init__(self)
        self.name = 'Trader'
        self.env = env
        self.action_dim = 4
        self.state_dim = 2886
        
    def reset(self):
        state = self.env.reset()
        state = np.array([state])
        #print('state')
        #print(state)
        return np.array(state)

    def step(self, action):
        #if action == 2:
            #print('action')
            #print(action)

        next_state, reward, done = self.env.step(action)
        #if reward != 0:
            #print(reward)
        next_state = np.array([next_state])
        reward = np.array([reward])
        if done:
            done = 1
            done = np.array([done])
            #print('done')
        else:
            done = np.array([done])
        #print('next_state')
        #print(next_state)
        #print('rewards')
        #print(reward)
        #print('done')
        #print(done)
        if np.any(done):
            next_state = self.reset()
        return next_state, reward, done, None
    
    def seed(self, random_seed):
        pass

def ppo_ppo_test():
    config = Config()
    #config.num_workers = 1
    #task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    task_fn = lambda: Trader()
    #task_fn = lambda log_dir: Trader()
    config.num_workers = 1
    #config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.task_fn = task_fn
    config.eval_env = task_fn()
    #config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * 5 * 10
    #config.max_steps = 1440
    config.logger = get_logger()
    return run_steps_2(PPOAgent(config))

success, rwd_deque, rwd_all, model_path = ppo_ppo_test()


if success:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rwd_all)), rwd_all)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if success:
    print("SUCCESS: Mean return of %.2f obtained over the last %d episodes. Victory was achieved after playing a total of %d episodes." % (
        np.mean(rwd_deque), len(rwd_deque), len(rwd_all)))



