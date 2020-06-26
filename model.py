import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weight_init
from shared_adam import SharedAdam


class ActorCritic(nn.Module):
    """actor-critic learner
    coupling the A and C
    Actor network(policy)
    returns policy function Pi(s,theta) obtained from actor NN
    Policy is given as a gaussian prob distribution for all actions
    with mean lying in [-1, 1] and sigma lying in (0,1)
    (Using Tanh() as normalization)

    Args:
        state_dim
        action_dim
        action_bound(int)
    """
    def __init__(self, 
                 state_dim,
                 action_dim,
                 action_bound):
        super(ActorCritic, self).__init__()
        self.action_bound = action_bound
        hidden_size = 512

        # Nested networks
        # Critic
        self.c1 = nn.Linear(state_dim, hidden_size)
        self.c1.weight.data.normal_(0.,0.1) 
        self.c2 = nn.Linear(hidden_size, hidden_size//2)
        self.c2.weight.data.normal_(0.,0.1) 
        self.c3 = nn.Linear(hidden_size//2, 1)
        self.c3.weight.data.normal_(0.,0.1) 
        
        # Actor
        self.a_1 = nn.Linear(state_dim, hidden_size)
        self.a_1.weight.data.normal_(0.,0.1) 
        self.a_2 = nn.Linear(hidden_size, hidden_size//2)
        self.a_2.weight.data.normal_(0.,0.1) 

        # Gaussian Distribution
        self.mu = nn.Linear(hidden_size//2, hidden_size//4)
        self.mu.weight.data.normal_(0.,0.1) 
        self.sigma = nn.Linear(hidden_size//2, hidden_size//4)
        self.sigma.weight.data.normal_(0.,0.1) 
        self.mu_out = nn.Linear(hidden_size//4, action_dim)
        self.mu_out.weight.data.normal_(0.,0.1) 
        self.sigma_out = nn.Linear(hidden_size//4, action_dim)
        self.sigma_out.weight.data.normal_(0.,0.1) 

    def forward(self, x):  
        out = F.relu(self.a_1(x))
        out = F.relu(self.a_2(out))
        out_mu = F.relu(self.mu(out))
        out_sigma = F.relu(self.sigma(out))

        mu = torch.tanh(self.mu_out(out_mu)) * self.action_bound # Norm
        sigma = F.softplus(self.sigma_out(out_sigma))  # Smoothing

        value = F.relu(self.c1(x))
        value = F.relu(self.c2(value))
        value = self.c3(value)

        return mu, sigma, value

