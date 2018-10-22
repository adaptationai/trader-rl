from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable

    
class DQN(nn.Module):
    def __init__(self, input_dims, output_dims, seed,  hidden_dim=64):
        """Initialize parameters and build model.
        Params
        ======
            input_dims (int): Dimension of each state
            output_dims (int): Dimension of each action
            seed (int): Random seed
            hidden_dim (int): Number of nodes in each hidden layer
            
        """
        super(DQN, self).__init__()
        # build model
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
       
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, output_dims)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
      
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dims, output_dims, seed,  hidden_dim=64, dueling_type="avg"):
        """Initialize parameters and build model.
        Params
        ======
            input_dims (int): Dimension of each state
            output_dims (int): Dimension of each action
            seed (int): Random seed
            hidden_dim (int): Number of nodes in each hidden layer
            dueling_type (string): Type of dueling network
            
        """
        super(DuelingDQN, self).__init__()
        # build model
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.dueling_type = dueling_type
        
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
     
        #Duleing [0]: V(s); [1,:]: A(s, a)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dims + 1)
        self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
        self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        
     
    def forward(self, x):
        x = x.view(x.size(0), self.input_dims)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x.view(x.size(0), -1))
        v_ind_vb = Variable(self.v_ind)
        a_ind_vb = Variable(self.a_ind)
        
        v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
        a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
        # now calculate Q(s, a)
        if self.dueling_type == "avg":      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
            x = v + (a - a.mean(1, keepdim=True))    
            
        elif self.dueling_type == "max":    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
            
            x = v + (a - a.max(1, keepdim=True)[0])
            
        elif self.dueling_type == "naive":  # Q(s,a)=V(s)+ A(s,a)
            x = v + a
                
        return x
        