import torch
import numpy as np
from collections import deque
from agent import Agent
from env import Template

env = Template("trader")


class DQN():
    # env assumption: env.reset(), env.render(), env.step(), env.close()
    def __init__(self, name, state_size, action_size, env, load_net=False):
        self.agent = Agent(state_size=state_size, action_size=action_size, seed=0, network="cnn")
        self.env = env
        self.saved_network = name+'_dqn_checkpoint.pth'
        self.load_net = load_net
        if load_net:
            print('Loading pretrained network...')
            self.agent.qnetwork_local.load_state_dict(torch.load(self.saved_network))
            self.agent.qnetwork_target.load_state_dict(torch.load(self.saved_network))
            print('Loaded.')

    def train(self, n_episodes=100000, max_t=1440, eps_start=1.0,
              eps_end=0.01, eps_decay=0.995,
              score_window_size=100, target_score=100.0,
              save=True,
              verbose=True):
        """Deep Q-Learning.

            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=score_window_size)  # last score_window_size scores
        eps = eps_start  # initialize epsilon
        save12 = False
        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                #if reward != 0:
                    #print(reward)
                if done:
                    #print('done')
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            avg_score = np.mean(scores_window)
            if avg_score>100 and not save12 and not self.load_net:
                torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                np.save('scores13_0824.npy', np.array(scores))
                save12 = True
            if avg_score >= target_score and i_episode>100:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                             np.mean(scores_window)))
                self.solved = True
                if save:
                    torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                break
            if i_episode % 1 == 0:
                torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                np.save('scores13_0824.npy', np.array(scores))

            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if save:
            torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)

        return scores

    def play(self, trials=100, steps=1440, load=True):
        if load:
            self.agent.qnetwork_local.load_state_dict(torch.load(self.saved_network))

        for i in range(trials):
            total_reward = 0
            print('Start Trial...')
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state)
                self.env.render()
                state, reward, done = self.env.step(action)
                total_reward += reward
                if reward!=0:
                    print("Current Reward:", reward, "Total Reward:", total_reward)
                if done:
                    print('Done.')
                    break
        #self.env.close()

training = DQN("name", 5766, 4, env, load_net=False)
training.train()
#training.play()