import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box, Discrete
from typing import Callable

from core.utils import match_to_list

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        module.bias.data.fill_(0)

""" Inverse dynamics model that is responsible for firstly extracting features from 
the raw observations and using those features to predict the the action taken by the 
agent given the current and following features. """
class InverseModel(nn.Module):
    def __init__(self, obs_dim, act_dim, in_channels, out_channels, 
                 kernel_sizes, strides, padding, hidden_sizes):
        super().__init__()
        self.act_dim = act_dim
        self.img_dim = obs_dim[-2:]
        self.in_channels = in_channels

        # Make sure that length of kernel_sizes, strides and padding are equal to out_channels
        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        if not isinstance(out_channels, list): out_channels = [out_channels]
        kernel_sizes = match_to_list(kernel_sizes, out_channels)
        strides = match_to_list(strides, out_channels)
        padding = match_to_list(padding, out_channels)

        # Initialize CNN feature extractor
        self.feature_ext = nn.Sequential()
        channels = [in_channels] + out_channels
        for i in range(len(kernel_sizes)):
            self.feature_ext.add_module(f'inv_fe_conv_{i+1}', nn.Conv2d(channels[i], channels[i+1],
                                                                        kernel_size=kernel_sizes[i],
                                                                        stride=strides[i],
                                                                        padding=padding[i]))
            self.feature_ext.add_module(f'inv_fe_act_{i+1}', nn.ELU())
        self.feature_ext.add_module('inv_fe_flatten', nn.Flatten())

        # Determine number of output features based on observation space
        self.feature_dim = self.feature_ext(torch.randn((1, in_channels, *self.img_dim), 
                                                        dtype=torch.float32)).shape[1]

        # Initialize hidden layers
        self.net = nn.Sequential()
        hidden_sizes = [2 * self.feature_dim] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(f'inv_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                               hidden_sizes[i+1]))
            self.net.add_module(f'inv_activation_{i+1}', nn.ReLU())
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Add output layer and initialize it
        self.net.add_module('inv_output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=0.01))
    
    def forward(self, obs):
        # obs shape (batch_size, seq_len, obs_dim)
        features = self.feature_ext(obs.flatten(0, 1))
        features = features.view(*obs.shape[:2], self.feature_dim)
        features_curr = features[:, :-1].flatten(0, 1)
        features_next = features[:, 1:].flatten(0, 1)

        act_pred = self.net(torch.cat([features_curr, features_next], dim=-1))
        act_pred = act_pred.view(features.shape[0], features.shape[1]-1, self.act_dim)

        return act_pred, features
    
    def layer_summary(self):
        x = torch.randn((1, self.in_channels, *self.img_dim))
        for layer in self.feature_ext:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        x = torch.randn((1, 2 * self.feature_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

""" Forward dynamics model that is responsible for estimating the next feature vector given 
the current features and the actual action taken by the agent between the two states. """
class ForwardModel(nn.Module):
    def __init__(self, act_transform: Callable, feature_dim, act_dim, hidden_sizes):
        super().__init__()
        self.act_transform = act_transform
        self.feature_dim = feature_dim
        self.act_dim = act_dim

        # Initialize hidden layers
        self.net = nn.Sequential()
        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        hidden_sizes = [feature_dim + act_dim] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(f'fwd_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                               hidden_sizes[i+1]))
            self.net.add_module(f'fwd_activation_{i+1}', nn.ReLU())
        
        # Add output layer and initialize all linear layers
        self.net.add_module('fwd_output', nn.Linear(hidden_sizes[-1], feature_dim))
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, features, act):
        # features shape (batch_size, seq_len, feature_dim)
        # act shape (batch_size, seq_len, act_dim)
        features_curr = features.flatten(0, 1)
        act_curr = self.act_transform(act).flatten(0, 1)

        features_next_pred = self.net(torch.cat([features_curr, act_curr], dim=-1))
        features_next_pred = features_next_pred.view(*features.shape[:2], self.feature_dim)
        
        return features_next_pred
    
    def layer_summary(self):
        x = torch.randn((1, self.feature_dim + self.act_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

"""Intrinsic Curiosity Module (ICM) that combines both the inverse and forward models. """
class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, env: VectorEnv, beta, in_channels, out_channels, kernel_sizes, 
                 strides, padding, hidden_size_inv, hidden_sizes_fwd):
        super().__init__()
        self.beta_2 = beta/2

        # Extract needed parameters from environment and actor-critic
        obs_dim = env.single_observation_space.shape
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            act_transform = lambda act: F.one_hot(act.long(), num_classes=act_dim)
        elif isinstance(env.single_action_space, Box):
            act_dim = env.single_action_space.shape[0]
            act_transform = lambda act: act

        # Initialize the inverse and forward models
        self.inv_mod = InverseModel(obs_dim, act_dim, in_channels, out_channels,
                                    kernel_sizes, strides, padding, hidden_size_inv)
        self.fwd_mod = ForwardModel(act_transform, self.inv_mod.feature_dim, 
                                    act_dim, hidden_sizes_fwd)
    
    def calc_reward(self, obs, obs_next, act):
        with torch.no_grad():
            # obs shape (batch_size, obs_dim)
            features = self.inv_mod.feature_ext(obs)
            features_next = self.inv_mod.feature_ext(obs_next)

            # intrinsic reward = beta/2 * SSD of feature prediction error
            features_next_pred = self.fwd_mod.forward(features.unsqueeze(1), act.unsqueeze(1))
            intrinsic_rew = self.beta_2 * torch.sum((features_next_pred.squeeze(1) - features_next)**2, dim=-1)

            return intrinsic_rew.cpu().numpy()
        
    def layer_summary(self):
        print('Inverse Model Summary: \n')
        self.inv_mod.layer_summary()

        print('Forward Model Summary: \n')
        self.fwd_mod.layer_summary()