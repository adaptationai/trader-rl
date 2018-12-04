import time

import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2
from ..env import Template_Gym
env = Template_Gym()

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")
                    
class CustomPolicy_2(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")

class CustomPolicy_3(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")


def make_env(env_id, rank, seed=0):
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

env_id = "CartPole-v1"
num_cpu = 16  # Number of processes to use
# Create the vectorized environment
#env = DummyVecEnv([lambda: env])

env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
                   
model = PPO2(CustomPolicy_2, env, verbose=0,tensorboard_log="./ppo2full10-3/" )
#model = PPO2(MlpLnLstmPolicy, env, verbose=0,tensorboard_log="./ppo2full10-3/" )
#model = PPO2(CustomPolicy, env, verbose=0, learning_rate=1e-6, tensorboard_log="./ppo2full10-2/" )
#model = DQN(CustomPolicy_3, env, verbose=0, prioritized_replay=True, tensorboard_log="./ppo2full10-2/" )
#model = ACKTR(CustomPolicy, env, verbose=0, learning_rate=1e-6, tensorboard_log="./ppo2full10-2/" )
#model = PPO2.load('trader256x3', env, policy=CustomPolicy, tensorboard_log="./ppo2full10/" )
#model.load("traderlstm.pkl")
#mode = PPO2.load('traderlstm2', env, verbose=0)
#model = PPO2(MlpLnLstmPolicy, env, verbose=1)
#model = ACKTR(MlpPolicy, env, nprocs=num_cpu)
#model.learn(total_timesteps=int(num_timesteps * 1.1), seed=seed)


def evaluate(model, num_steps=14400):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
      # _states are only useful when using LSTM policies
      actions, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(actions)
      
      # Stats
      for i in range(env.num_envs):
          episode_rewards[i][-1] += rewards[i]
          if dones[i]:
              episode_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])     
        n_episodes += len(episode_rewards[i])   

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


#mean_reward_before_train = evaluate(model, num_steps=14400)

n_timesteps = 100000000

# Multiprocessed RL Training
start_time = time.time()
#model.learn(total_timesteps=int(n_timesteps * 1.1), seed=0)
model.learn(n_timesteps)
total_time_multi = time.time() - start_time
model.save("trader256x3-1_full")

print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

mean_reward_before_train = evaluate(model, num_steps=144000)