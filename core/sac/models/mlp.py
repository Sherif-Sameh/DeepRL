from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        module.bias.data.fill_(0)

def polyak_average(params, target_params, polyak):
    with torch.no_grad():
        for param, param_target in zip(params, target_params):
            param_target.data.mul_(polyak)
            param_target.data.add_(param.data, alpha=1-polyak)

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
        hidden_sizes = [obs_dim] + hidden_sizes
        self.net = nn.Sequential()
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(prefix + f'_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                                     hidden_sizes[i+1]))
            self.net.add_module(prefix + f'_activation_{i+1}', hidden_acts[i]())

        # Initialize all parameters of hidden layers
        self.net.apply(lambda m: init_weights(m, gain=1.0))
    
    def layer_summary(self):
        x = torch.randn((1, self.obs_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

    def forward(self, obs):
        return self.net(obs)


class MLPCritic(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts):
        # Initialize MLP hidden layers
        super().__init__('critic', obs_dim + act_dim, hidden_sizes, hidden_acts)

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('critic_output', nn.Linear(hidden_sizes[-1], 1))
        self.net[-1].apply(lambda m: init_weights(m, gain=1.0))

        # Create and initialize target Q network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False
    
    def forward(self, obs, act):
        q = self.net(torch.cat([obs, act], dim=-1))

        return q
    
    def forward_target(self, obs, act):
        with torch.no_grad():
            q = self.net_target(torch.cat([obs, act], dim=-1))

        return q
    
    def update_target(self, polyak):
        polyak_average(self.net.parameters(), self.net_target.parameters(), polyak)

    def set_grad_tracking(self, val: bool):
        for param in self.net.parameters():
            param.requires_grad = val


class MLPActor(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts, 
                 action_max):
        # Initialize MLP hidden layers
        super().__init__('actor', obs_dim, hidden_sizes, hidden_acts)
        self.act_dim = act_dim
        self.action_max = action_max

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('actor_output', nn.Linear(hidden_sizes[-1], 2 * act_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=0.01))
    
    def forward(self, obs):
        a, log_p = self.log_prob(obs)
        
        return a, log_p
    
    def log_prob(self, obs):
        # Input is assumed to be always 2D
        out = self.net(obs)
        mu, std = out[:, :self.act_dim], torch.exp(out[:, self.act_dim:])
        
        # Compute the actions 
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)

        # Compute log probability
        normal, eps = Normal(loc=mu, scale=std), 1e-6
        log_p = normal.log_prob(u).sum(dim=-1) - torch.log(1 - a**2 + eps).sum(dim=-1)

        return a * self.action_max, log_p
    
    def log_prob_no_grad(self, obs):
        with torch.no_grad():
            a, log_p = self.log_prob(obs)
        
        return a, log_p
    

class MLPActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic):
        super().__init__()
        self.action_max = nn.Parameter(torch.tensor(env.single_action_space.high), 
                                       requires_grad=False)

        # Check the action space type and initialize the actor
        if isinstance(env.single_action_space, Box):
            obs_dim = env.single_observation_space.shape[0]
            act_dim = env.single_action_space.shape[0]
            self.actor = MLPActor(obs_dim, act_dim, hidden_sizes_actor, 
                                  hidden_acts_actor, self.action_max)
        else:
            raise NotImplementedError
        
        # Initialize the two critics (DQNs)
        self.critic_1 = MLPCritic(obs_dim, act_dim, hidden_sizes_critic,
                               hidden_acts_critic)
        self.critic_2 = MLPCritic(obs_dim, act_dim, hidden_sizes_critic,
                               hidden_acts_critic)
        
    def step(self, obs):
        with torch.no_grad():
            act, log_prob = self.actor.forward(obs)
            q_val_1 = self.critic_1.forward(obs, act)
            q_val_2 = self.critic_2.forward(obs, act)
            q_val = torch.min(q_val_1, q_val_2)

        return act.cpu().numpy(), q_val.cpu().numpy().squeeze(), log_prob.cpu().numpy()
    
    def act(self, obs):
        with torch.no_grad():
            out = self.actor.net(obs)
            act = out[..., :self.actor.act_dim] # Take the mean of the SAC policy
        
        return act.cpu().numpy()
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        out = self.actor.net(obs)
        act = out[..., :self.actor.act_dim]
        q_val = self.critic_1.net(torch.cat([obs, act], dim=-1))

        return out, q_val
    
    def layer_summary(self):
        print('Actor Summary: \n')
        self.actor.layer_summary()

        print('Critic Summary: \n')
        self.critic_1.layer_summary()