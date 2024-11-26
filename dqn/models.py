import numpy as np
import gym
from gym.spaces import Discrete
from copy import deepcopy

import torch
from torch import nn

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        module.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, hidden_acts):
        super().__init__()
        self.obs_dim = obs_dim

        # Make sure that length of hidden_sizes and hidden_acts is the same
        if not isinstance(hidden_acts, list):
            hidden_acts = [hidden_acts] * len(hidden_sizes)
        elif len(hidden_acts) < len(hidden_sizes):
            hidden_acts = [hidden_acts[0]] * len(hidden_sizes)
        
        # Initialize all hidden layers
        self.net = nn.Sequential()
        hidden_sizes = [obs_dim] + hidden_sizes
        for i in range(len(hidden_acts)):
            self.net.add_module(f'hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                           hidden_sizes[i+1]))
            self.net.add_module(f'activation_{i+1}', hidden_acts[i]())

        # Initialize all parameters of hidden layers
        self.net.apply(lambda m: init_weights(m, gain=1.0))

    def layer_summary(self):
        x = torch.randn((1, self.obs_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

    def forward(self):
        raise NotImplementedError
    
    
class MLPDQN(MLP):
    def __init__(self, env: gym.Env, eps_init, eps_final, 
                 eps_decay_rate, hidden_sizes, hidden_acts):
        # Check action space type
        if isinstance(env.action_space, Discrete):
            act_dim = env.action_space.n
            obs_dim = env.observation_space.shape[0]
        else:
            raise NotImplementedError
        
        # Initialize MLP hidden layers
        super().__init__(obs_dim, hidden_sizes, hidden_acts)

        # Save epsilon scheduling parameters
        self.eps = eps_init
        self.eps_min = eps_final
        self.eps_decay_rate = eps_decay_rate

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=0.01))

        # Create and initialize target Q network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False

    def __eps_greedy(self, obs):
        with torch.no_grad():
            r1 = np.random.rand()
            q_vals = self.net(obs).squeeze()
            if r1 < self.eps:
                a = torch.randint(0, q_vals.shape[-1], size=(1,))
                q = q_vals[a]
            else:
                q, a = torch.max(q_vals, dim=-1)
        
        return a.item(), q.item()
    
    def forward_grad(self, obs, act):
        q_vals = self.net(obs)
        q = q_vals[torch.arange(q_vals.shape[0]), act]

        return q
    
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.net_target(obs)
            q, _ = torch.max(q_vals, dim=-1)
        
        return q
    
    def update_target(self):
        with torch.no_grad():
            for param, param_target in zip(self.net.parameters(), self.net_target.parameters()):
                param_target.data.copy_(param.data)

    def update_eps_exp(self):
        self.eps = max(self.eps_min, self.eps * np.exp(-self.eps_decay_rate))        
    
    def step(self, obs):
        act, q_val = self.__eps_greedy(obs)

        return act, q_val
    
    def act(self, obs):
        with torch.no_grad():
            q_vals = self.net(obs).squeeze()
            act = torch.argmax(q_vals, dim=-1)

        return act.numpy()


class DuelingQLayer(nn.Module):
    def __init__(self, num_hiddens, act_dim):
        super().__init__()
        self.v_layer = nn.Linear(num_hiddens, 1)
        self.adv_layer = nn.Linear(num_hiddens, act_dim)

        self.v_layer.apply(lambda m: init_weights(m, gain=1.0))
        self.adv_layer.apply(lambda m: init_weights(m, gain=0.01))

    def forward(self, H):
        v = self.v_layer(H)
        adv = self.adv_layer(H)

        return v + (adv - torch.mean(adv, dim=-1, keepdim=True))
    
class MLPDuelingDQN(MLPDQN):
    def __init__(self, env: gym.Env, eps_init, eps_final, 
                 eps_decay_rate, hidden_sizes, hidden_acts):
        # Check action space type
        if isinstance(env.action_space, Discrete):
            act_dim = env.action_space.n
            obs_dim = env.observation_space.shape[0]
        else:
            raise NotImplementedError
        
        # Initialize MLP hidden layers
        MLP.__init__(self, obs_dim, hidden_sizes, hidden_acts)

        # Save epsilon scheduling parameters
        self.eps = eps_init
        self.eps_min = eps_final
        self.eps_decay_rate = eps_decay_rate

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('Dueling_Out', DuelingQLayer(hidden_sizes[-1], act_dim))

        # Create and initialize target Q network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False
    
class MLPDDQN(MLPDQN):
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.net(obs)
            act = torch.argmax(q_vals, dim=-1)
            q_vals_target = self.net_target(obs)
            q = q_vals_target[torch.arange(q_vals.shape[0]), act]
        
        return q

class MLPDuelingDDQN(MLPDuelingDQN):
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.net(obs)
            act = torch.argmax(q_vals, dim=-1)
            q_vals_target = self.net_target(obs)
            q = q_vals_target[torch.arange(q_vals.shape[0]), act]
        
        return q