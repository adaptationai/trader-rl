
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

#from stable_baselines.deepq import DQN, MlpPolicy

#from env import Template
from template_env import Template
env = Template()

#env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
#model = PPO2.load("bl_pp02.pkl")
model = PPO2(MlpPolicy, env, verbose=0)
#model.load("bl_ppo2.pkl")
#model = model.load("bl_pp02.pkl")
#model = DQN(MlpPolicy, env, verbose=0)
#model = model.load("cartpole_model.pkl")
#model = DQN(
        #env=env,
        #policy=MlpPolicy,
        #learning_rate=1e-3,
        #buffer_size=50000,
        #exploration_fraction=0.1,
        #exploration_final_eps=0.02,
model.learn(total_timesteps=10000000)
model.save("bl_ppo2")
#model.save("bl_dqn")
#model = PPO2.load('bl_pp02')

#obs = env.reset()
#for i in range(10000):
    #action, _states = model.predict(obs)
    #action = 1
    #obs, rewards, dones, info = env.step(action)
    #print(f'obs: {obs}')
    #print(f'rewards: {rewards}')
    #print(f'dones: {dones}')
    #print(f'obs': {obs})
    #env.render()

def evaluate(model, num_steps=100000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += rewards[0]
      if dones[0]:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward
  
  return mean_100ep_reward
mean_reward_before_train = evaluate(model, num_steps=100000)
#model.learn(total_timesteps=10000)