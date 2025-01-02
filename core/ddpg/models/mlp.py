from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from copy import deepcopy

import torch
from torch import nn

from core.rl_utils import polyak_average

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
        # Initialize MLP hidden layers (except final hidden layer)
        super().__init__('critic', obs_dim, hidden_sizes[:-1], hidden_acts)
        self.act_dim = act_dim

        # Initialize final hidden layer (feed in actions)
        final_act = hidden_acts[-1] if isinstance(hidden_acts, list) else hidden_acts
        self.net.add_module(f'critic_hidden_{len(hidden_sizes)}', nn.Linear(hidden_sizes[-2] + act_dim, 
                                                                            hidden_sizes[-1]))
        self.net.add_module(f'critic_activation_{len(hidden_sizes)}', final_act())

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('critic_output', nn.Linear(hidden_sizes[-1], 1))
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
    
    # Used in CNN-AC to track gradients only after feature extraction
    def forward_actions(self, obs, act):
        return self.forward(obs, act)
    
    def forward_target(self, obs, act):
        with torch.no_grad():
            q = self.net_target[:-3](obs)
            q = torch.cat([q, act], dim=1)
            q = self.net_target[-3:](q)

        return q
    
    def update_target(self, polyak):
        polyak_average(self.net.parameters(), self.net_target.parameters(), polyak)
    
    def set_grad_tracking(self, val: bool):
        for param in self.net.parameters():
            param.requires_grad = val

    def layer_summary(self):
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
        

class MLPActor(MLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts, action_max):
        # Initialize MLP hidden layers
        super().__init__('actor', obs_dim, hidden_sizes, hidden_acts)
        self.action_max = action_max

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('actor_output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net.add_module('actor_output_act', nn.Tanh())
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
    
    def update_target(self, polyak):
        polyak_average(self.net.parameters(), self.net_target.parameters(), polyak)


class MLPActorCritic(nn.Module):
    def __init__(self, env: VectorEnv, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic, action_std, action_std_f):
        super().__init__()
        self.action_max = nn.Parameter(torch.tensor(env.single_action_space.high, dtype=torch.float32), 
                                       requires_grad=False)

        # Check the action space type and initialize the actor
        if isinstance(env.single_action_space, Box):
            obs_dim = env.single_observation_space.shape[0]
            act_dim = env.single_action_space.shape[0]
            self.actor = MLPActor(obs_dim, act_dim, hidden_sizes_actor, 
                                  hidden_acts_actor, self.action_max)
        else:
            raise NotImplementedError
        
        # Initialize the standard deviation of actions used for training
        if len(action_std) != act_dim: action_std = [action_std[0]] * act_dim
        if action_std_f[0] < 0: action_std_f = action_std
        if len(action_std_f) != act_dim: action_std_f = [action_std_f[0]] * act_dim
        
        action_std = torch.tensor(action_std, dtype=torch.float32)
        action_std_f = torch.tensor(action_std_f, dtype=torch.float32)
        self.action_std = nn.Parameter(action_std, requires_grad=False)
        self.action_std_rate = nn.Parameter(action_std - action_std_f, requires_grad=False)

        # Initialize the critic (DQN)
        self.critic = MLPCritic(obs_dim, act_dim, hidden_sizes_critic,
                             hidden_acts_critic)
        
    def step(self, obs):
        with torch.no_grad():
            act = self.actor.forward(obs)
            act = torch.max(torch.min(act + self.action_std * torch.randn_like(act), 
                                      self.action_max), -self.action_max)
            q_val = self.critic.forward(obs, act)

        return act.cpu().numpy(), q_val.cpu().numpy().squeeze()
    
    def act(self, obs):
        with torch.no_grad():
            act = self.actor.forward(obs)
        
        return act.cpu().numpy()
    
    # Empty method used only by LSTM actor-critics
    def reset_hidden_states(self, device, batch_size=1, batch_idx=None,
                            save=False, restore=False):
        pass
    
    # Only for tracing the actor and critic's networks for tensorboard
    def forward(self, obs):
        act = self.actor(obs)
        q_val = self.critic(obs, act)

        return act, q_val
    
    def step_action_std(self, epochs):
        self.action_std.data -= (self.action_std_rate.data)/(epochs+1)
    
    def layer_summary(self):
        print('Actor Summary: \n')
        self.actor.layer_summary()

        print('Critic Summary: \n')
        self.critic.layer_summary()