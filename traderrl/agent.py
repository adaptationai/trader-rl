import numpy as np
import random
from collections import namedtuple, deque
from memory import ReplayBuffer
from model import DQN
from model import DuelingDQN



import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR = 0.0005              # learning rate 
UPDATE_EVERY = 4        # how often to update the network
TARGET_UPDATE = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, network):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.network = network
        

        # Q-Network
        if self.network == "duel":
            self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
       
        else:
            self.qnetwork_local = DQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DQN(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
 
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, count):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, count)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    
    def learn(self, experiences, gamma, count):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        # Q values for best actions in next_state
        # from current Q network
        
        if self.network == "double" or "duel":
            Q_L = self.qnetwork_local(next_states).detach()
            _, actions_prime = Q_L.max(1)

        # get Q values from frozen network for next state and chosen action
        
        Q_targets_next = self.qnetwork_target(next_states).detach()
        Q_targets_next_s_a_prime = Q_targets_next.gather(1, actions_prime.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next_s_a_prime * (1 - dones))

        # Get expected Q values from target model using current actions
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        #if count >= TARGET_UPDATE:
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
  
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


