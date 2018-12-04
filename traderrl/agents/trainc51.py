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
from ..lib import deep_rl
#from deep_rl import *
import torch
import numpy as np
from collections import deque
#from agent import Agent
from ..env import Template_Gym
#from template_env import Template_Gym

def run_steps_2(agent):
    random_seed()
    config = agent.config
    save_path = '1000_update_trader_placement1.bin'
    agent.load(save_path)
    while True:
        rewards = agent.rewards_deque
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):            
            config.logger.info('Mean return of %.2f over a window of %d recent episodes. %d episodes played so far, most recent episodic return is %.2f' % (
                np.mean(rewards), len(rewards), len(agent.rewards_all), agent.rewards_all[-1]))
            save_path = '1000_update_trader_placement1.bin'
            agent.save(save_path)

        if len(rewards) >= 100 and agent.rewards_all[-1] % 100 == 0:
            save_path = '1000_update_trader_placement1.bin'
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

class ClassicalControl_2(BaseTask):
    def __init__(self, name='CartPole-v0', max_steps=1440, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = Template_Gym()
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        #self.env = self.set_monitor(self.env, log_dir)

def categorical_dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    task_fn = lambda: ClassicalControl_2()
    #task_fn = lambda log_dir: Trader()
    config.num_workers = 3
    #config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.task_fn = task_fn

    config.eval_env = task_fn()
    #config.task_fn = ClassicalControl_2()

    #config.eval_env = ClassicalControl_2()
    
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, FCBody(config.state_dim))
    #config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    #config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32)
    #config.replay_fn = lambda: AsyncReplay(memory_size=1000000, batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(28800), batch_size=64)

    config.discount = 0.99
    config.target_network_update_freq = 5760
    config.exploration_steps = 300000
    config.categorical_v_max = 200
    config.categorical_v_min = -200
    config.categorical_n_atoms = 51
    config.gradient_clip = 5
    config.sgd_update_frequency = 4
    config.save_interval = 14400
    config.eval_interval = int(14400)
    #config.max_steps = 1e7
    config.async_actor = True
    config.max_steps = 1e7
    config.logger = get_logger()
    #config.logger = get_logger(tag=categorical_dqn_pixel_atari.__name__)
    run_steps(CategoricalDQNAgent(config))
success, rwd_deque, rwd_all, model_path = categorical_dqn_cart_pole()


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



