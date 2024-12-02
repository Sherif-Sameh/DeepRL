import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box, Discrete

import torch
from torch import nn
import torch.distributions
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        module.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, prefix, obs_dim, hidden_sizes, hidden_acts):
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
            self.net.add_module(prefix + f'_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                           hidden_sizes[i+1]))
            self.net.add_module(prefix + f'_activation_{i+1}', hidden_acts[i]())

        # Initialize all parameters of hidden layers
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def layer_summary(self):
        x = torch.randn((1, self.obs_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

    def forward(self, obs):
        raise NotImplementedError
    
    
class MLPActor(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts):
        super().__init__('actor', obs_dim, hidden_sizes, hidden_acts)

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('actor_output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=0.01))

        # Initialize a generic stochastic policy
        self.pi = torch.distributions.Distribution(validate_args=False)

    def forward(self, obs):
        raise NotImplementedError
    
    def update_policy(self, obs):
        raise NotImplementedError
    
    def log_prob_no_grad(self, act):
        raise NotImplementedError
    
    def log_prob_grad(self, obs, act):
        raise NotImplementedError
    
    def surrogate_obj(self, obs, act, adv, log_prob_prev):
        raise NotImplementedError

    def kl_divergence_grad(self):
        raise NotImplementedError
    
    def kl_divergence_no_grad(self, pi_prev):
        raise NotImplementedError

    
class MLPActorDiscrete(MLPActor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts):
        super().__init__(obs_dim, act_dim, hidden_sizes, hidden_acts)
        
        # Initialize the policy randomly
        self.pi = Categorical(logits=torch.randn(act_dim, dtype=torch.float32))

    def forward(self, obs):
        with torch.no_grad():
            logits = self.net(obs)
            self.pi = Categorical(logits=logits)
            a = self.pi.sample()
        
        return a

    def update_policy(self, obs):
        with torch.no_grad():
            logits = self.net(obs)
            self.pi = Categorical(logits=logits)

    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act)
        
        return log_prob
    
    def log_prob_grad(self, obs, act):
        logits = self.net(obs)
        self.pi = Categorical(logits=logits)
        
        return self.pi.log_prob(act)
    
    def surrogate_obj(self, obs, act, adv, log_prob_prev):
        self.update_policy(obs) # update policy after parameter update
        log_prob = self.log_prob_no_grad(act).numpy()
        
        return np.mean(np.exp(log_prob - log_prob_prev) * adv)
    
    def kl_divergence_grad(self):
        # Note: self.pi should've been already updated by log_prob_grad()
        with torch.no_grad():
            logits = self.pi.logits.detach()
            pi = Categorical(logits=logits)
        
        return torch.distributions.kl_divergence(pi, self.pi).mean()
    
    def kl_divergence_no_grad(self, pi_prev):
        with torch.no_grad():
            kl = torch.distributions.kl.kl_divergence(pi_prev, self.pi)
        
        return kl.mean()
            
class MLPActorContinuous(MLPActor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts):
        super().__init__(obs_dim, act_dim, hidden_sizes, hidden_acts)
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float32)
        self.log_std = nn.Parameter(log_std, requires_grad=True)

        # Initialize the policy randomly
        self.pi = Normal(loc=torch.randn(act_dim), scale=torch.exp(self.log_std))

    def forward(self, obs):
        with torch.no_grad():
            mean = self.net(obs)
            self.pi = Normal(mean, torch.exp(self.log_std))
            a = self.pi.sample()
        
        return a
    
    def update_policy(self, obs):
        with torch.no_grad():
            mean = self.net(obs)
            self.pi = Normal(mean, torch.exp(self.log_std))

    def log_prob_no_grad(self, act):
        with torch.no_grad():
            log_prob = self.pi.log_prob(act).sum(axis=-1)
            
        return log_prob
    
    def log_prob_grad(self, obs, act):
        mean = self.net(obs)
        self.pi = Normal(mean, torch.exp(self.log_std))

        return self.pi.log_prob(act).sum(axis=-1)
    
    def surrogate_obj(self, obs, act, adv, log_prob_prev):
        self.update_policy(obs) # update policy after parameter update
        log_prob = self.log_prob_no_grad(act).numpy()

        return np.mean(np.exp(log_prob - log_prob_prev) * adv)
    
    def kl_divergence_grad(self):
        # Note: self.pi should've been already updated by log_prob_grad()
        mu1, log_std1 = self.pi.mean, self.log_std
        var1 = torch.exp(2 * log_std1)
        with torch.no_grad():
            mu0, log_std0 = self.pi.mean.detach(), self.log_std.data
            var0 = torch.exp(2 * log_std0)

        # Compute the element-wise KL divergence
        pre_sum = 0.5 * (((mu0 - mu1) ** 2 + var0) / (var1 + 1e-8) - 1) + log_std1 - log_std0
        
        # Sum over the dimensions of the distribution
        all_kls = torch.sum(pre_sum, dim=1)
        
        # Return the mean KL divergence over the batch
        return torch.mean(all_kls)
    
    def kl_divergence_no_grad(self, pi_prev):
        mu1, log_std1 = self.pi.mean.detach(), self.log_std.data
        var1 = torch.exp(2 * log_std1)
        with torch.no_grad():
            mu0, var0 = pi_prev.mean.detach(), pi_prev.variance.detach()
            log_std0 = torch.log(pi_prev.stddev)
        
        # Compute the element-wise KL divergence
        pre_sum = 0.5 * (((mu0 - mu1) ** 2 + var0) / (var1 + 1e-8) - 1) + log_std1 - log_std0
        
        # Sum over the dimensions of the distribution
        all_kls = torch.sum(pre_sum, dim=1)
        
        # Return the mean KL divergence over the batch
        return torch.mean(all_kls)
    
class MLPCritic(MLP):
    def __init__(self, obs_dim, hidden_sizes, hidden_acts):
        super().__init__('critic', obs_dim, hidden_sizes, hidden_acts)
        
        # Add the output layer to the network and intialize its weights 
        self.net.add_module('critic_output', nn.Linear(hidden_sizes[-1], 1))
        self.net[-1].apply(lambda m: init_weights(m, gain=1))

    def forward(self, obs):
        with torch.no_grad():
            v = torch.squeeze(self.net(obs), -1)
        
        return v
    
    def forward_grad(self, obs):
        return torch.squeeze(self.net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]

        # Initialize the actor based on the action space type of the env
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            self.actor = MLPActorDiscrete(obs_dim, act_dim, hidden_sizes_actor, 
                                          hidden_acts_actor)
        elif isinstance(env.single_action_space, Box):
            act_dim = env.single_action_space.shape[0]
            self.actor = MLPActorContinuous(obs_dim, act_dim, hidden_sizes_actor, 
                                            hidden_acts_actor)
        else:
            raise NotImplementedError
        
        # Initialize the critic
        self.critic = MLPCritic(obs_dim, hidden_sizes_critic,
                                hidden_acts_critic)
        
    def step(self, obs):
        act = self.actor(obs)
        val = self.critic(obs)
        logp = self.actor.log_prob_no_grad(act)

        return act.numpy(), val.numpy(), logp.numpy()
    
    def act(self, obs):
        return self.actor(obs).numpy()
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        act_net = self.actor.net(obs)
        val_net = self.critic.net(obs)

        return act_net, val_net
    
    def layer_summary(self):
        print('Actor Summary: \n')
        self.actor.layer_summary()

        print('Critic Summary: \n')
        self.critic.layer_summary()