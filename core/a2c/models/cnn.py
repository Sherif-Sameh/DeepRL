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

class CNNActor(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor, act_dim):
        super().__init__()
        self.feature_ext = feature_ext
        self.act_dim = act_dim

        # Add the actor's output layer and intialize its weights 
        self.actor_head = nn.Sequential(OrderedDict([
            ('actor_output', nn.Linear(feature_ext.features_out, act_dim))
            ]))
        self.actor_head.apply(lambda m: init_weights(m, gain=0.01))

    @abc.abstractmethod
    def forward(self, obs):
        """Updates the actor's policy using given observations, then samples 
        and returns actions from the updated polciy. Always evaluates the 
        network under torch.no_grad(). """
        pass
    
    @abc.abstractmethod
    def log_prob_no_grad(self, act):
        """Evaluates and returns log probabilities of the given actions 
        with respect to the current stored policy. Always evaluates 
        probabilites under torch.no_grad(). """
        pass
    
    @abc.abstractmethod
    def log_prob_grad(self, obs, act):
        """Re-evaluates the network and updates the actor's policy using 
        the given observations. Then, it evaluates the log probabilities
        of the given actions from the policy with gradient tracking enabled.
        Returns the computed log probabilites. """
        pass

    def layer_summary(self):
        print(self.actor_head[0].__class__.__name__, 'input & output shapes:\t', 
              f'(1, {self.feature_ext.features_out})', f'(1, {self.act_dim})\n')

class CNNActorDiscrete(CNNActor):
    def __init__(self, feature_ext: FeatureExtractor, act_dim):
        super().__init__(feature_ext, act_dim)
        
        # Initialize the policy randomly
        self.pi = Categorical(logits=torch.randn(act_dim, dtype=torch.float32))

    def forward(self, obs):
        with torch.no_grad():
            logits = self.actor_head(self.feature_ext(obs))
            self.pi = Categorical(logits=logits)
            a = self.pi.sample()
        
        return a

    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act.squeeze())
        
        return log_prob
    
    def log_prob_grad(self, obs, act):
        logits = self.actor_head(self.feature_ext(obs))
        self.pi = Categorical(logits=logits)

        return self.pi.log_prob(act.squeeze())
    
class CNNActorContinuous(CNNActor):
    def __init__(self, feature_ext: FeatureExtractor, act_dim):
        super().__init__(feature_ext, act_dim)
        log_std = -1.0 * torch.ones(act_dim, dtype=torch.float32)
        self.log_std = nn.Parameter(log_std, requires_grad=True)

        # Initialize the policy randomly
        self.pi = Normal(loc=torch.randn(act_dim), scale=torch.exp(self.log_std))

    def forward(self, obs):
        with torch.no_grad():
            mean = self.actor_head(self.feature_ext(obs))
            self.pi = Normal(mean, torch.exp(self.log_std))
            a = self.pi.sample()
        
        return a

    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act).sum(axis=-1)
        
        return log_prob
    
    def log_prob_grad(self, obs, act):
        mean = self.actor_head(self.feature_ext(obs))
        self.pi = Normal(mean, torch.exp(self.log_std))

        return self.pi.log_prob(act).sum(axis=-1)
    
class CNNCritic(nn.Module):
    def __init__(self, feature_ext: FeatureExtractor):
        super().__init__()
        self.feature_ext = feature_ext

        # Add the critic's output layer and intialize its weights 
        self.critic_head = nn.Sequential(OrderedDict([
            ('critic_output', nn.Linear(feature_ext.features_out, 1))
            ]))
        self.critic_head.apply(lambda m: init_weights(m, gain=1))
        
    def forward(self, obs):
        with torch.no_grad():
            v = torch.squeeze(self.critic_head(self.feature_ext(obs)))
        
        return v
    
    def forward_grad(self, obs):
        return torch.squeeze(self.critic_head(self.feature_ext(obs)))

    def layer_summary(self):
        print(self.critic_head[0].__class__.__name__, 'input & output shapes:\t', 
              f'(1, {self.feature_ext.features_out})', '(1, 1)\n')
    
class CNNActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, in_channels, out_channels, 
                 kernel_sizes, strides, features_out):
        super().__init__()
        obs_dim = env.single_observation_space.shape

        # Determine action dimension from environment
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            actor_type = CNNActorDiscrete
        elif isinstance(env.single_action_space, Box):
            act_dim = env.single_action_space.shape[0]
            actor_type = CNNActorContinuous
        else:
            raise NotImplementedError
        
        # Initialize shared feature extractor as well as actor and critic
        self.feature_ext = FeatureExtractor(obs_dim, in_channels, out_channels,
                                            kernel_sizes, strides, features_out)
        self.actor = actor_type(self.feature_ext, act_dim)
        self.critic = CNNCritic(self.feature_ext)
    
    def step(self, obs):
        act = self.actor(obs)
        logp = self.actor.log_prob_no_grad(act)
        val = self.critic(obs)

        return act.cpu().numpy(), val.cpu().numpy(), logp.cpu().numpy()
    
    def act(self, obs):
        # Make sure that obs always has a batch dimension
        if obs.ndim == 3:
            return self.actor(obs[None]).squeeze(dim=0).cpu().numpy()
        
        return self.actor(obs).cpu().numpy()
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        features = self.feature_ext(obs)
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
        