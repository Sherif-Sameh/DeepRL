from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from .mlp import init_weights, polyak_average

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, in_channels, out_channels, 
                 kernel_sizes, strides, features_out):
        super().__init__()
        self.img_dim = obs_dim[-2:]
        self.in_channels = in_channels
        self.features_out = features_out[-1] if isinstance(features_out, list) else features_out

        # Make sure that length of kernel_sizes and strides are equal to out_channels
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * len(out_channels)
            strides = [strides] * len(out_channels)
        elif len(kernel_sizes) < len(out_channels):
            kernel_sizes = [kernel_sizes[0]] * len(out_channels)
            strides = [strides[0]] * len(out_channels)

        # Initialize all convolutional layers
        self.net = nn.Sequential()
        channels = [in_channels] + out_channels
        for i in range(len(kernel_sizes)):
            self.net.add_module(f'fe_conv_{i+1}', nn.Conv2d(channels[i], channels[i+1], 
                                                            kernel_size=kernel_sizes[i],
                                                            stride=strides[i], padding=0))
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


class CNNCritic(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor, feature_ext_target:FeatureExtractor,
                 act_dim, hidden_sizes):
        super().__init__()
        self.feature_ext = feature_ext
        self.feature_ext_target = feature_ext_target
        self.act_dim = act_dim

        # Gather critic's independent layers
        layers = []
        hidden_sizes = [hidden_sizes] if not isinstance(hidden_sizes, list) else hidden_sizes
        hidden_sizes = [feature_ext.features_out + act_dim] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.append((f'critic_hidden_{i+1}', nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            layers.append((f'critic_activation_{i+1}', nn.ReLU()))
        layers.append(('critic_output', nn.Linear(hidden_sizes[-1], 1)))

        # Initialize the critic's head and its weights
        self.critic_head = nn.Sequential(OrderedDict(layers))
        self.critic_head.apply(lambda m: init_weights(m, gain=1.0))

        # Create and initialize target critic's head
        self.critic_head_target = deepcopy(self.critic_head)
        for param in self.critic_head_target.parameters():
            param.requires_grad = False

    def forward(self, obs, act):
        return self.critic_head(torch.cat([self.feature_ext(obs), act], dim=1))
    
    def forward_target(self, obs, act):
        with torch.no_grad():
            q = self.critic_head_target(torch.cat([self.feature_ext_target(obs), act], dim=1))
        
        return q
    
    def update_target(self, polyak):
        polyak_average(self.feature_ext.parameters(), self.feature_ext_target.parameters(), polyak)
        polyak_average(self.critic_head.parameters(), self.critic_head_target.parameters(), polyak)

    def set_grad_tracking(self, val: bool):
        for param in self.critic_head.parameters():
            param.requires_grad = val

    def layer_summary(self):
        x = torch.randn((1, self.feature_ext.features_out + self.act_dim), dtype=torch.float32)
        for layer in self.critic_head:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')


class CNNActor(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor, feature_ext_target: FeatureExtractor, 
                 act_dim, hidden_sizes, action_max):
        super().__init__()
        self.feature_ext = feature_ext
        self.feature_ext_target = feature_ext_target
        self.action_max = action_max

        # Gather actor's independent layers
        layers = []
        hidden_sizes = [hidden_sizes] if not isinstance(hidden_sizes, list) else hidden_sizes
        hidden_sizes = [feature_ext.features_out] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.append((f'actor_hidden_{i+1}', nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            layers.append((f'actor_activation_{i+1}', nn.ReLU()))
        layers.append(('actor_output', nn.Linear(hidden_sizes[-1], act_dim)))
        layers.append(('actor_output_act', nn.Tanh()))

        # Initialize the actor's head and its weights
        self.actor_head = nn.Sequential(OrderedDict(layers))
        self.actor_head[:-2].apply(lambda m: init_weights(m, gain=1.0))
        self.actor_head[-2:].apply(lambda m: init_weights(m, gain=0.01))

        # Create and initialize target actor's head
        self.actor_head_target = deepcopy(self.actor_head)
        for param in self.actor_head_target.parameters():
            param.requires_grad = False
    
    def forward(self, obs):
        return self.actor_head(self.feature_ext(obs)) * self.action_max
    
    def forward_target(self, obs):
        with torch.no_grad():
            a = self.actor_head_target(self.feature_ext_target(obs)) * self.action_max
        
        return a

    def update_target(self, polyak):
        polyak_average(self.feature_ext.parameters(), self.feature_ext_target.parameters(), polyak)
        polyak_average(self.actor_head.parameters(), self.actor_head_target.parameters(), polyak)
    
    def layer_summary(self):
        x = torch.randn((1, self.feature_ext.features_out), dtype=torch.float32)
        for layer in self.actor_head:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')
    
class CNNActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, in_channels, out_channels, 
                 kernel_sizes, strides, features_out, hidden_sizes_actor, 
                 hidden_sizes_critic, action_std):
        super().__init__()
        self.action_std = action_std
        self.action_max = nn.Parameter(torch.tensor(env.single_action_space.high), 
                                       requires_grad=False)

        # Check the action space type and initialize the actor
        if isinstance(env.single_action_space, Box):
            obs_dim = env.single_observation_space.shape
            act_dim = env.single_action_space.shape[0]
        else:
            raise NotImplementedError
        
        # Initialize the feature extractors, critics and actor
        self.feature_ext = FeatureExtractor(obs_dim, in_channels, out_channels, 
                                            kernel_sizes, strides, features_out)
        self.feature_ext_target = deepcopy(self.feature_ext)
        for param in self.feature_ext_target.parameters(): 
            param.requires_grad = False
        self.actor = CNNActor(self.feature_ext, self.feature_ext_target, 
                              act_dim, hidden_sizes_actor, self.action_max)
        self.critic_1 = CNNCritic(self.feature_ext, self.feature_ext_target, 
                                  act_dim, hidden_sizes_critic)
        self.critic_2 = CNNCritic(self.feature_ext, self.feature_ext_target, 
                                  act_dim, hidden_sizes_critic)
        
    def step(self, obs):
        with torch.no_grad():
            features = self.feature_ext(obs)
            act = self.actor.actor_head(features)
            act = torch.max(torch.min(act + self.action_std * torch.randn_like(act),
                                      self.action_max), -self.action_max)
            q_val = self.critic_1.critic_head(torch.cat([features, act], dim=1))
        
        return act.cpu().numpy(), q_val.cpu().numpy().squeeze()

    def act(self, obs):
        with torch.no_grad():
            if obs.ndim == 3:
                act = self.actor.forward(obs[None]).squeeze(dim=0)
            else:
                act = self.actor.forward(obs)
        
        return act.cpu().numpy()
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        features = self.feature_ext(obs)
        act = self.actor.actor_head(features)
        q_val = self.critic_1.critic_head(torch.cat([features, act], dim=1))

        return act, q_val
    
    def layer_summary(self):
        print('Feature Extractor Summary: \n')
        self.feature_ext.layer_summary()

        print('Actor Head: \n')
        self.actor.layer_summary()

        print('Critic Head: \n')
        self.critic_1.layer_summary()