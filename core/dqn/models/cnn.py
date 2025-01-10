import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Discrete
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from .mlp import init_weights, DuelingQLayer
from core.utils import match_to_list

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, in_channels, out_channels, kernel_sizes, 
                 strides, padding, features_out):
        super().__init__()
        self.img_dim = obs_dim[-2:]
        self.in_channels = in_channels
        self.features_out = features_out[-1] if isinstance(features_out, list) else features_out

        # Make sure that length of kernel_sizes, strides and padding are equal to out_channels
        if not isinstance(out_channels, list): out_channels = [out_channels]
        kernel_sizes = match_to_list(kernel_sizes, out_channels)
        strides = match_to_list(strides, out_channels)
        padding = match_to_list(padding, out_channels)

        # Initialize all convolutional layers
        self.net = nn.Sequential()
        channels = [in_channels] + out_channels
        for i in range(len(kernel_sizes)):
            self.net.add_module(f'fe_conv_{i+1}', nn.Conv2d(channels[i], channels[i+1], 
                                                            kernel_size=kernel_sizes[i],
                                                            stride=strides[i], 
                                                            padding=padding[i]))
            self.net.add_module(f'fe_act_{i+1}', nn.ReLU())
        self.net.add_module('fe_flatten', nn.Flatten())

        # Determine number of output features based on observation space
        fe_out = self.net(torch.randn((1, in_channels, *self.img_dim), dtype=torch.float32))

        # Add linear layers and initialize them
        features_out = [features_out] if not isinstance(features_out, list) else features_out
        features_out = [fe_out.shape[1]] + features_out
        self.linear = nn.Sequential()
        for i in range(len(features_out)-1):
            self.linear.add_module(f'fe_fc_hidden_{i+1}', nn.Linear(features_out[i], features_out[i+1]))
            self.linear.add_module(f'fe_fc_act_{i+1}', nn.ReLU())
        self.linear.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(self, obs):
        return self.linear(self.net(obs))
    
    def layer_summary(self):
        x = torch.randn((1, self.in_channels, *self.img_dim), dtype=torch.float32)
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        for layer in self.linear:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')


class CNNDQN(nn.Module):
    def __init__(self, env: VectorEnv, eps_init, eps_final, 
                 eps_decay_rate, in_channels, out_channels, 
                 kernel_sizes, strides, padding, features_out):
        # Check action space type
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            obs_dim = env.single_observation_space.shape
        else:
            raise NotImplementedError
        
        # Initialize feature extractor
        super().__init__()
        self.feature_ext = FeatureExtractor(obs_dim, in_channels, out_channels, kernel_sizes,
                                            strides, padding, features_out)
        self.act_dim = act_dim
        
        # Save epsilon scheduling parameters
        self.eps = eps_init
        self.eps_min = eps_final
        self.eps_decay_rate = eps_decay_rate
        
        # Initialize the Q-network head and its weights
        self.q_head = nn.Sequential(OrderedDict([
            ('dqn_output', nn.Linear(self.feature_ext.features_out, act_dim))
            ]))
        self.q_head.apply(lambda m: init_weights(m, gain=1.0))

        # Create and initialize target Q network
        self.feature_ext_target = deepcopy(self.feature_ext)
        self.q_head_target = deepcopy(self.q_head)
        for param in self.feature_ext_target.parameters():
            param.requires_grad = False
        for param in self.q_head_target.parameters():
            param.requires_grad = False

    def __eps_greedy(self, obs):
        with torch.no_grad():
            r1 = torch.rand(size=(obs.shape[0],), device=obs.device)
            q_vals = self.q_head(self.feature_ext(obs))
            
            # Random actions
            rand_a = torch.randint(0, q_vals.shape[-1], size=(obs.shape[0],), device=obs.device)
            
            # Greedy actions
            greedy_a = torch.argmax(q_vals, dim=-1)

            # Choose between them based on r1
            a = torch.where(r1 < self.eps, rand_a, greedy_a)
            q = q_vals[torch.arange(q_vals.shape[0]), a]
        
        return a, q
    
    # Only for tracing the network for tensorboard
    def forward(self, obs):
        features = self.feature_ext(obs)
        q_vals = self.q_head(features)
        
        return q_vals
    
    def forward_grad(self, obs, act):
        q_vals = self.q_head(self.feature_ext(obs))
        q = q_vals[torch.arange(q_vals.shape[0]), act]

        return q
    
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.q_head_target(self.feature_ext_target(obs))
            q, _ = torch.max(q_vals, dim=-1)
        
        return q
    
    def update_target(self):
        self.feature_ext_target.load_state_dict(self.feature_ext.state_dict())
        self.q_head_target.load_state_dict(self.q_head.state_dict())

    def update_eps_exp(self):
        self.eps = max(self.eps_min, self.eps * np.exp(-self.eps_decay_rate))

    def step(self, obs):
        act, q_val = self.__eps_greedy(obs)

        return act.cpu().numpy(), q_val.cpu().numpy()  

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            # Make sure that obs always has a batch dimension
            if obs.ndim == 3:
                q_vals = self.q_head(self.feature_ext(obs[None])).squeeze(dim=0)
            else:
                q_vals = self.q_head(self.feature_ext(obs))
            act = torch.argmax(q_vals, dim=-1)

        return act.cpu().numpy()    

    def layer_summary(self):
        print('Feature Extractor Summary: \n')
        self.feature_ext.layer_summary()

        print('Q-Net Head: \n')
        print(self.q_head[0].__class__.__name__, 'input & output shapes:\t', 
              f'(1, {self.feature_ext.features_out})', f'(1, {self.act_dim})')
        
class CNNDuelingDQN(CNNDQN):
    def __init__(self, env: VectorEnv, eps_init, eps_final, 
                 eps_decay_rate, in_channels, out_channels, 
                 kernel_sizes, strides, features_out):
        super().__init__(env, eps_init, eps_final, eps_decay_rate, 
                         in_channels, out_channels, kernel_sizes,
                         strides, features_out)
        
        # Override the regular q_head with a dueling layer
        self.q_head = nn.Sequential(OrderedDict([
            ('dueling_output', DuelingQLayer(self.feature_ext.features_out, self.act_dim))
            ]))
        self.q_head.apply(lambda m: init_weights(m, gain=1.0))

        # Re-copy target q_head and disable gradients
        self.q_head_target = deepcopy(self.q_head)
        for param in self.q_head_target.parameters():
            param.requires_grad = False
    
class CNNDDQN(CNNDQN):
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.q_head(self.feature_ext(obs))
            act = torch.argmax(q_vals, dim=-1)
            q_vals_target = self.q_head_target(self.feature_ext_target(obs))
            q = q_vals_target[torch.arange(q_vals.shape[0]), act]
        
        return q

class CNNDuelingDDQN(CNNDuelingDQN):
    def forward_target(self, obs):
        with torch.no_grad():
            q_vals = self.q_head(self.feature_ext(obs))
            act = torch.argmax(q_vals, dim=-1)
            q_vals_target = self.q_head_target(self.feature_ext_target(obs))
            q = q_vals_target[torch.arange(q_vals.shape[0]), act]
        
        return q