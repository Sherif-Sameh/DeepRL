import abc
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box, Discrete
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from .mlp import init_weights
from .cnn import CNNActor as CNNLSTMActor

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, in_channels, out_channels, kernel_sizes, strides, features_out):
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
    
    """ Forward propogate through a single batch index. Needed for terminal value 
    calculation at the end episode when truncated. """
    def forward_batch_idx(self, obs, batch_idx):
        # Evaluate Conv. layers on flattened input sequences
        fe_out = self.net(obs.flatten(0, 1))

        # Reshape output and evaluate LSTMs on specific batch index
        fe_out = fe_out.view(*obs.shape[:2], fe_out.shape[-1])
        for i, layer in enumerate(self.lstm):
            fe_out, (h, c) = layer(fe_out, (self.lstm_h[i][:, batch_idx:batch_idx+1], 
                                            self.lstm_c[i][:, batch_idx:batch_idx+1]))
            self.lstm_h[i][0, batch_idx] = h.detach().clone().squeeze() 
            self.lstm_c[i][0, batch_idx] = c.detach().clone().squeeze()

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

class CNNLSTMActorDiscrete(CNNLSTMActor):
    def __init__(self, feature_ext: FeatureExtractor, act_dim):
        super().__init__(feature_ext, act_dim)
        
        # Initialize the policy randomly
        self.pi = Categorical(logits=torch.randn(act_dim, dtype=torch.float32))

    def forward(self, obs, features=None, deterministic=False):
        if features is None: features = self.feature_ext(obs)
        logits = self.actor_head(features.flatten(0, 1)).view(*features.shape[:2], self.act_dim)
        self.pi = Categorical(logits=logits)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            a = self.pi.sample()
        
        return a

    def copy_policy(self):
        return Categorical(logits=self.pi.logits)

    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act)
        
        return log_prob
    
    def log_prob_grad(self, obs, act):
        logits = self.actor_head(self.feature_ext(obs).flatten(0, 1))
        self.pi = Categorical(logits=logits.view(*obs.shape[:2], self.act_dim))

        return self.pi.log_prob(act)
    
    def kl_divergence(self, obs, pi_prev: Categorical):
        obs, mask = obs # Unpack observation and episode mask tuple
        features = self.feature_ext(obs)
        logits = self.actor_head(features.flatten(0, 1)).view(*features.shape[:2], self.act_dim)
        self.pi = Categorical(logits=logits)
        kl = torch.distributions.kl.kl_divergence(pi_prev, self.pi)
        
        return kl[mask].mean()

    def entropy(self):
        return self.pi.entropy()
    
class CNNLSTMActorContinuous(CNNLSTMActor):
    def __init__(self, feature_ext: FeatureExtractor, act_dim, log_std_init):
        super().__init__(feature_ext, act_dim)

        # Initialize policy log std
        if len(log_std_init) != act_dim:
            log_std_init = [log_std_init[0]] * act_dim
        log_std = torch.tensor(log_std_init, dtype=torch.float32)
        self.log_std = nn.Parameter(log_std, requires_grad=True)

        # Initialize the policy randomly
        self.pi = Normal(loc=torch.randn(act_dim), scale=torch.exp(self.log_std))

    def forward(self, obs, features=None, deterministic=False):
        if features is None: features = self.feature_ext(obs)
        mean = self.actor_head(features.flatten(0, 1)).view(*features.shape[:2], self.act_dim)
        self.pi = Normal(mean, torch.exp(self.log_std))
        if deterministic:
            a = mean
        else:
            a = self.pi.sample()

        return a

    def copy_policy(self):
        return Normal(loc=self.pi.mean, scale=self.pi.stddev)
    
    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act).sum(axis=-1)
        
        return log_prob
    
    def log_prob_grad(self, obs, act):
        mean = self.actor_head(self.feature_ext(obs).flatten(0, 1))
        self.pi = Normal(mean.view(*obs.shape[:2], self.act_dim), 
                         torch.exp(self.log_std))

        return self.pi.log_prob(act).sum(axis=-1)
    
    def kl_divergence(self, obs, pi_prev: Normal):
        obs, mask = obs # Unpack observation and episode mask tuple
        features = self.feature_ext(obs)
        mean = self.actor_head(features.flatten(0, 1)).view(*features.shape[:2], self.act_dim)
        self.pi = Normal(mean, torch.exp(self.log_std))
        kl = torch.distributions.kl.kl_divergence(pi_prev, self.pi)

        return kl[mask].mean()
    
    def entropy(self):
        return self.pi.entropy().sum(dim=-1)
    
class CNNLSTMCritic(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor):
        super().__init__()
        self.feature_ext = feature_ext

        # Add the critic's output layer and intialize its weights 
        self.critic_head = nn.Sequential(OrderedDict([
            ('critic_output', nn.Linear(feature_ext.features_out, 1))
            ]))
        self.critic_head.apply(lambda m: init_weights(m, gain=1))
        
    def forward(self, obs, features=None):
        if features is None: features = self.feature_ext(obs)
        v = torch.squeeze(self.critic_head(features.flatten(0, 1)))
        v = v.view(*features.shape[:2])

        return v

    def layer_summary(self):
        print(self.critic_head[0].__class__.__name__, 'input & output shapes:\t', 
              f'(1, {self.feature_ext.features_out})', '(1, 1)\n')

class CNNLSTMActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, in_channels, out_channels, 
                 kernel_sizes, strides, features_out, log_std_init):
        super().__init__()
        obs_dim = env.single_observation_space.shape

        # Initialize shared feature extractor
        self.feature_ext = FeatureExtractor(obs_dim, in_channels, out_channels,
                                            kernel_sizes, strides, features_out)

        # Determine action dimension from environment and initialize actor
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            self.actor = CNNLSTMActorDiscrete(self.feature_ext, act_dim)
        elif isinstance(env.single_action_space, Box):
            act_dim = env.single_action_space.shape[0]
            self.actor = CNNLSTMActorContinuous(self.feature_ext, act_dim, 
                                                log_std_init)
        else:
            raise NotImplementedError
        
        # Initialize critic
        self.critic = CNNLSTMCritic(self.feature_ext)

        # Initialize empty variable for storing hidden state during training
        self.hidden_state_stored = None
    
    def step(self, obs):
        with torch.no_grad(): 
            features = self.feature_ext.forward(obs.unsqueeze(1))
            act = self.actor.forward(obs, features=features)
            val = self.critic.forward(obs, features=features)
            logp = self.actor.log_prob_no_grad(act)

        return act.squeeze(1).cpu().numpy(), val.squeeze(1).cpu().numpy(), \
            logp.squeeze(1).cpu().numpy()
    
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
    
    def get_terminal_value(self, obs, batch_idx):
        with torch.no_grad():
            features = self.feature_ext.forward_batch_idx(obs[batch_idx].unsqueeze(0).unsqueeze(0), 
                                                          batch_idx)
            val_term = self.critic.critic_head(features.squeeze(0, 1)).squeeze()

        return val_term.cpu().numpy()
    
    # Clears LSTM hidden states across a single or all batch indices
    def reset_hidden_states(self, device, batch_size=1, batch_idx=None,
                            save=False, restore=False):
        if batch_idx is None:
            if batch_idx is None:
                if restore == True:
                    for i, (c0, h0) in enumerate(zip(*self.hidden_state_stored)):
                        self.feature_ext.lstm_c[i] = c0.clone()
                        self.feature_ext.lstm_h[i] = h0.clone()
                    return
                if save == True:
                    self.hidden_state_stored = [[], [], [], []]
                    for c0, h0 in zip(self.feature_ext.lstm_c, self.feature_ext.lstm_h):
                        self.hidden_state_stored[0].append(c0.clone())
                        self.hidden_state_stored[1].append(h0.clone())
                    return
            
            self.feature_ext.reset_hidden_state_all(device, batch_size=batch_size)
        else:
            self.feature_ext.reset_hidden_state(device, batch_idx=batch_idx)
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        self.reset_hidden_states(torch.device('cpu'), batch_size=obs.shape[0])
        features = self.feature_ext(obs.unsqueeze(1)).squeeze(1)
        act = self.actor.actor_head(features)
        val = self.critic.critic_head(features)

        return act, val
    
    def layer_summary(self):
        print('Feature Extractor Summary: \n')
        self.feature_ext.layer_summary()

        print('Actor Head: \n')
        self.actor.layer_summary()

        print('Critic Head: \n')
        self.critic.layer_summary()