from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

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

        # Add all LSTM layers
        features_out = [features_out] if not isinstance(features_out, list) else features_out
        features_out = [fe_out.shape[1]] + features_out
        self.lstm = nn.Sequential()
        for i in range(len(features_out)-1):
            self.lstm.add_module(f'fe_lstm_{i+1}', nn.LSTM(features_out[i], features_out[i+1],
                                                           batch_first=True))    
            
        # Store the number of hidden units in each LSTM layer and initialize their hidden states
        self.lstm_hiddens = features_out[1:]
        self.lstm_h = [None] * len(self.lstm_hiddens)
        self.lstm_c = [None] * len(self.lstm_hiddens)
        self.reset_hidden_state_all(device=torch.device('cpu'))

    def forward(self, obs):
        # Evaluate Conv. layers on flattened input sequences
        fe_out = self.net(obs.flatten(0, 1))

        # Reshape output and evaluate LSTMs
        fe_out = fe_out.view(*obs.shape[:2], fe_out.shape[-1])
        for i, layer in enumerate(self.lstm):
            fe_out, (h, c) = layer(fe_out, (self.lstm_h[i], self.lstm_c[i]))
            self.lstm_h[i], self.lstm_c[i] = h.detach().clone(), c.detach().clone()

        return fe_out

    """ Resets a specific batch index for all hidden states across all LSTM layers. Typically 
    used for resetting the hidden state of a specific environment during rollouts. """ 
    def reset_hidden_state(self, device, batch_idx):
        for i, hidden_size in enumerate(self.lstm_hiddens):
            self.lstm_h[i][0, batch_idx] = torch.zeros(hidden_size, dtype=torch.float32,
                                                       device=device)
            self.lstm_c[i][0, batch_idx] = torch.zeros(hidden_size, dtype=torch.float32,
                                                       device=device)

    """ Re-initializes all hidden states across all LSTM layers with the given batch size. 
    Typically used before loss computation and at the end of an epoch. """             
    def reset_hidden_state_all(self, device, batch_size=1):
        for i, hidden_size in enumerate(self.lstm_hiddens):
            self.lstm_h[i] = torch.zeros((1, batch_size, hidden_size), dtype=torch.float32,
                                         device=device)
            self.lstm_c[i] = torch.zeros((1, batch_size, hidden_size), dtype=torch.float32,
                                         device=device)

    def layer_summary(self):
        x = torch.randn((1, self.in_channels, *self.img_dim), dtype=torch.float32)
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        x = x.reshape(1, 1, -1)
        for i, layer in enumerate(self.lstm):
            input_shape = x.shape
            h0 = torch.zeros((1, 1, self.lstm_hiddens[i]), dtype=torch.float32)
            c0 = torch.zeros((1, 1, self.lstm_hiddens[i]), dtype=torch.float32)
            x, _ = layer(x, (h0, c0))
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')


class CNNLSTMCritic(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor, feature_ext_target:FeatureExtractor,
                 act_dim, hidden_sizes):
        super().__init__()
        self.feature_ext = feature_ext
        self.feature_ext_target = feature_ext_target
        self.act_dim = act_dim

        # Gather critic's independent layers
        layers = []
        hidden_sizes = [] if hidden_sizes is None else hidden_sizes
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

    def forward(self, obs, act, features=None):
        if features is None: features = self.feature_ext(obs)
        q = self.critic_head(torch.cat([features, act], dim=2).flatten(0, 1))
        q = q.view(*features.shape[:2], 1)

        return q
    
    # Used in CNN-AC to track gradients only after feature extraction
    def forward_actions(self, obs, act):
        with torch.no_grad():
            features = self.feature_ext(obs)
        q = self.critic_head(torch.cat([features, act], dim=2).flatten(0, 1))
        q = q.view(*obs.shape[:2], 1)
        
        return q
    
    def forward_target(self, obs, act):
        with torch.no_grad():
            features = self.feature_ext_target(obs)
            q = self.critic_head_target(torch.cat([features, act], dim=2).flatten(0, 1))
            q = q.view(*obs.shape[:2], 1)
        
        return q
    
    def update_target(self, polyak):
        polyak_average(self.feature_ext.parameters(), self.feature_ext_target.parameters(), np.sqrt(polyak))
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


class CNNLSTMActor(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor, act_dim, 
                 hidden_sizes, action_max):
        super().__init__()
        self.feature_ext = feature_ext
        self.act_dim = act_dim
        self.action_max = action_max
        self.act_dim = act_dim

        # Gather actor's independent layers
        layers = []
        hidden_sizes = [] if hidden_sizes is None else hidden_sizes
        hidden_sizes = [hidden_sizes] if not isinstance(hidden_sizes, list) else hidden_sizes
        hidden_sizes = [feature_ext.features_out] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.append((f'actor_hidden_{i+1}', nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            layers.append((f'actor_activation_{i+1}', nn.ReLU()))
        layers.append(('actor_output', nn.Linear(hidden_sizes[-1],  2 * act_dim)))

        # Initialize the actor's head and its weights
        self.actor_head = nn.Sequential(OrderedDict(layers))
        self.actor_head.apply(lambda m: init_weights(m, gain=1.0))
        self.actor_head[-1:].apply(lambda m: init_weights(m, gain=0.01))
    
    def forward(self, obs, features=None, deterministic=False):
        # Input is assumed to be always 5D
        if features is None: features = self.feature_ext(obs) 
        out = self.actor_head(features.flatten(0, 1))
        mu, std = out[:, :self.act_dim], torch.exp(out[:, self.act_dim:])
        mu, std = mu.view(*features.shape[:2], self.act_dim), std.view(*features.shape[:2], self.act_dim)
        
        # Compute the actions 
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)

        return a * self.action_max
    
    def log_prob(self, obs, features=None):
        # Input is assumed to be always 5D
        if features is None: features = self.feature_ext(obs) 
        out = self.actor_head(features.flatten(0, 1))
        mu, std = out[:, :self.act_dim], torch.exp(out[:, self.act_dim:])
        mu, std = mu.view(*features.shape[:2], self.act_dim), std.view(*features.shape[:2], self.act_dim)
        
        # Compute the actions 
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)

        # Compute log probability
        normal, eps = Normal(loc=mu, scale=std), 1e-6
        log_p = normal.log_prob(u).sum(dim=-1) - torch.log(1 - a**2 + eps).sum(dim=-1)

        return a * self.action_max, log_p
    
    def log_prob_no_grad(self, obs, features=None):
        with torch.no_grad():
            a, log_p = self.log_prob(obs, features=features)
        
        return a, log_p
    
    def layer_summary(self):
        x = torch.randn((1, self.feature_ext.features_out), dtype=torch.float32)
        for layer in self.actor_head:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')
    
class CNNLSTMActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, in_channels, out_channels, kernel_sizes, 
                 strides, features_out, hidden_sizes_actor, hidden_sizes_critic):
        super().__init__()
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
        self.actor = CNNLSTMActor(self.feature_ext, act_dim, hidden_sizes_actor, 
                                  self.action_max)
        self.critic_1 = CNNLSTMCritic(self.feature_ext, self.feature_ext_target, 
                                      act_dim, hidden_sizes_critic)
        self.critic_2 = CNNLSTMCritic(self.feature_ext, self.feature_ext_target, 
                                      act_dim, hidden_sizes_critic)
        
        # Initialize empty variable for storing hidden state during training
        self.hidden_state_stored = None
        
    def step(self, obs):
        with torch.no_grad():
            features = self.feature_ext(obs.unsqueeze(1))
            act = self.actor.forward(obs, features=features)
            q_val_1 = self.critic_1.forward(obs, act, features=features)
            q_val_2 = self.critic_2.forward(obs, act, features=features)
            q_val = torch.min(q_val_1, q_val_2)
        
        return act.squeeze(1).cpu().numpy(), q_val.cpu().numpy().squeeze()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            # Make sure that obs always has batch and sequence dimensions
            if obs.ndim == 3:
                act = self.actor.forward(obs.unsqueeze(0).unsqueeze(0),
                                         deterministic=deterministic).squeeze(0, 1)
            else:
                act = self.actor.forward(obs.unsqueeze(1),
                                         deterministic=deterministic).squeeze(1)
                        
        return act.cpu().numpy()
    
    # Clears LSTM hidden states across a single or all batch indices
    def reset_hidden_states(self, device, batch_size=1, batch_idx=None, 
                            save=False, restore=False):
        if batch_idx is None:
            if restore == True:
                for i, (c0, h0, c0_target, h0_target) in enumerate(zip(*self.hidden_state_stored)):
                    self.feature_ext.lstm_c[i] = c0.clone()
                    self.feature_ext.lstm_h[i] = h0.clone()
                    self.feature_ext_target.lstm_c[i] = c0_target.clone()
                    self.feature_ext_target.lstm_h[i] = h0_target.clone()
                return
            if save == True:
                self.hidden_state_stored = [[], [], [], []]
                for c0, h0, c0_target, h0_target in zip(self.feature_ext.lstm_c, self.feature_ext.lstm_h,
                                                        self.feature_ext_target.lstm_c, self.feature_ext_target.lstm_h):
                    self.hidden_state_stored[0].append(c0.clone())
                    self.hidden_state_stored[1].append(h0.clone())
                    self.hidden_state_stored[2].append(c0_target.clone())
                    self.hidden_state_stored[3].append(h0_target.clone())
                return
                
            self.feature_ext.reset_hidden_state_all(device, batch_size=batch_size)
            self.feature_ext_target.reset_hidden_state_all(device, batch_size=batch_size)
        else:
            self.feature_ext.reset_hidden_state(device, batch_idx=batch_idx)
            self.feature_ext_target.reset_hidden_state(device, batch_idx=batch_idx)
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        self.reset_hidden_states(torch.device('cpu'), batch_size=obs.shape[0])
        features = self.feature_ext(obs.unsqueeze(1))
        out = self.actor.actor_head(features.squeeze(1))
        act = out[..., :self.actor.act_dim]
        q_val = self.critic_1.forward(obs, act.unsqueeze(1), features=features)

        return out, q_val
    
    def layer_summary(self):
        print('Feature Extractor Summary: \n')
        self.feature_ext.layer_summary()

        print('Actor Head: \n')
        self.actor.layer_summary()

        print('Critic Head: \n')
        self.critic_1.layer_summary()