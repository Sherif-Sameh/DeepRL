import numpy as np
import gym
from gym.spaces import Box
from copy import deepcopy

import torch
from torch import nn

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        module.bias.data.fill_(0)

def polyak_average(params, target_params, polyak):
    with torch.no_grad():
        for param, param_target in zip(params, target_params):
            param_target.data.mul_(polyak)
            param_target.data.add_(param.data, alpha=1-polyak)

class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, hidden_acts, use_BN=False):
        super().__init__()
        self.obs_dim = obs_dim

        # Make sure that length of hidden_sizes and hidden_acts is the same
        if not isinstance(hidden_acts, list):
            hidden_acts = [hidden_acts] * len(hidden_sizes)
        elif len(hidden_acts) < len(hidden_sizes):
            hidden_acts = [hidden_acts[0]] * len(hidden_sizes)

        # Initialize all hidden layers
        hidden_sizes = [obs_dim] + hidden_sizes
        self.net = nn.Sequential()
        if use_BN == True:
            self.net.add_module(f'batch_norm_0', nn.BatchNorm1d(obs_dim))
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(f'hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                           hidden_sizes[i+1]))
            if use_BN == True:
                self.net.add_module(f'batch_norm_{i+1}', nn.BatchNorm1d(hidden_sizes[i+1]))
            self.net.add_module(f'activation_{i+1}', hidden_acts[i]())

        # Initialize all parameters of hidden layers
        self.net.apply(lambda m: init_weights(m, gain=1.0))
    
    def layer_summary(self):
        self.net.eval()
        x = torch.randn((1, self.obs_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')
        self.net.train()


class MLPDQN(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts, use_BN=False):
        # Initialize MLP hidden layers (except final hidden layer)
        super().__init__(obs_dim, hidden_sizes[:-1], hidden_acts, use_BN=use_BN)
        self.act_dim = act_dim

        # Initialize final hidden layer (feed in actions)
        final_act = hidden_acts[-1] if isinstance(hidden_acts, list) else hidden_acts
        self.net.add_module(f'hidden_{len(hidden_sizes)}', nn.Linear(hidden_sizes[-2] + act_dim, 
                                                           hidden_sizes[-1]))
        self.net.add_module(f'activation_{len(hidden_sizes)}', final_act())

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('output', nn.Linear(hidden_sizes[-1], 1))
        self.net[-3:].apply(lambda m: init_weights(m, gain=1.0))

        # Create and initialize target Q network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False
    
    def forward(self, obs, act):
        q = self.net[:-3](obs)
        q = torch.cat([q, act], dim=1)
        q = self.net[-3:](q)

        return q
    
    def forward_target(self, obs, act):
        with torch.no_grad():
            q = self.net_target[:-3](obs)
            q = torch.cat([q, act], dim=1)
            q = self.net_target[-3:](q)

        return q
    
    def set_grad_tracking(self, val: bool):
        for param in self.net.parameters():
            param.requires_grad = val

    def layer_summary(self):
        self.net.eval()
        x = torch.randn((1, self.obs_dim))
        for layer in self.net[:-3]:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        
        x = torch.cat([x, torch.randn((1, self.act_dim))], dim=1)
        for layer in self.net[-3:]:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')
        self.net.train()
        

class MLPActor(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts, 
                 action_max, use_BN=False):
        # Initialize MLP hidden layers
        self.action_max = action_max
        super().__init__(obs_dim, hidden_sizes, hidden_acts, use_BN=use_BN)

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net.add_module('output_act', nn.Tanh())
        self.net[-2:].apply(lambda m: init_weights(m, gain=0.01))

        # Create and initialize target actor network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False
    
    def forward(self, obs):
        return self.net(obs) * self.action_max
    
    def forward_target(self, obs):
        with torch.no_grad():
            a = self.net_target(obs) * self.action_max

        return a


class MLPActorCritic(nn.Module):
    def __init__(self, env: gym.Env, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic, action_std, use_BN=False):
        super().__init__()
        self.action_std = action_std
        self.action_max = torch.tensor(env.action_space.high) 

        # Check the action space type and initialize the actor
        if isinstance(env.action_space, Box):
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            self.actor = MLPActor(obs_dim, act_dim, hidden_sizes_actor, 
                                  hidden_acts_actor, self.action_max, use_BN=use_BN)
        else:
            raise NotImplementedError
        
        # Initialize the critic (DQN)
        self.critic = MLPDQN(obs_dim, act_dim, hidden_sizes_critic,
                             hidden_acts_critic, use_BN=use_BN)
        
    def step(self, obs):
        with torch.no_grad():
            obs = torch.unsqueeze(obs, dim=0)
            act = self.actor.forward(obs)
            act = torch.max(torch.min(act + self.action_std * torch.randn_like(act), 
                                      self.action_max), -self.action_max)
            q_val = self.critic.forward(obs, act)

        return act.numpy().squeeze(axis=0), q_val.numpy().squeeze(axis=0)
    
    def act(self, obs):
        with torch.no_grad():
            act = self.actor.forward(torch.unsqueeze(obs, dim=0))
        
        return act.numpy().squeeze(axis=0)
    
    def layer_summary(self):
        print('Actor Summary: \n')
        self.actor.layer_summary()

        print('Critic Summary: \n')
        self.critic.layer_summary()