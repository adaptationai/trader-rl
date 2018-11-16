import argparse

import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq import DQN, MlpPolicy
from ..env import Template_Gym
env = Template_Gym()
env = DummyVecEnv([lambda: env])

from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[512,512],
                                           layer_norm=True,
                                            feature_extraction="mlp")
class Baseline_Details():
    def __init__(self):
        self.love = "Ramona"

    def callback(self, lcl, _glb):
        """
        The callback function for logging and saving
        :param lcl: (dict) the local variables
        :param _glb: (dict) the global variables
        :return: (bool) is solved
        """
        # stop training if reward exceeds 199
        if len(lcl['episode_rewards'][-101:-1]) == 0:
            mean_100ep_reward = -np.inf
        else:
            mean_100ep_reward = round(float(np.mean(lcl['episode_rewards'][-101:-1])), 1)
        is_solved = lcl['step'] > 100 and mean_100ep_reward >= 100
        return is_solved


    def main(self, args):
        """
        Train and save the DQN model, for the cartpole problem
        :param args: (ArgumentParser) the input arguments
        """
    

        #env = gym.make('CartPole-v1')
        #model = DQN(MlpPolicy, env, verbose=1)
        #model.load("cartpole_model.pkl")
        model = DQN(
            env=env,
            policy=CustomPolicy,
            learning_rate=1e-3,
            buffer_size=50000,
            exploration_fraction=0.01,
            exploration_final_eps=0.02,
            verbose=1
        )
        model.learn(total_timesteps=args.max_timesteps, callback=self.callback)

        print("Saving model to cartpole_model.pkl")
        model.save("cartpole_model.pkl")


#if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    #parser.add_argument('--max-timesteps', default=100000000, type=int, help="Maximum number of timesteps")
    #args = parser.parse_args()
#main(args)