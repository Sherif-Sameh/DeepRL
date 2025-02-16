from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from core.utils import match_to_list
from core.rl_utils import copy_parameters

from core.sac.models.mlp import init_weights, MLP
from core.sac.models.mlp import MLPActor as SACMLPActor
from core.sac.models.mlp import MLPCritic as SACMLPCritic
from core.sac.models.mlp import MLPActorCritic as SACMLPActorCritic

class RegMLP(MLP):
    def __init__(self, prefix, obs_dim, hidden_sizes, hidden_acts, dropout_prob):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim

        # Make sure that length of hidden_sizes and hidden_acts is the same
        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        hidden_acts = match_to_list(hidden_acts, hidden_sizes)

        # Initialize all hidden layers
        hidden_sizes = [obs_dim] + hidden_sizes
        self.net = nn.Sequential()
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(prefix + f'_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                                     hidden_sizes[i+1]))
            if dropout_prob > 0:
                self.net.add_module(prefix + f'_dropout_{i+1}', nn.Dropout(p=dropout_prob))
            self.net.add_module(prefix + f'_layernorm_{i+1}', nn.LayerNorm(hidden_sizes[i+1]))
            self.net.add_module(prefix + f'_activation_{i+1}', hidden_acts[i]())

        # Initialize all parameters of hidden layers
        self.net.apply(lambda m: init_weights(m, gain=1.0))

class RegSACMLPCritic(SACMLPCritic, RegMLP):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hidden_acts, dropout_prob):
        # Initialize Regularized MLP
        RegMLP.__init__(self, 'critic', obs_dim + act_dim, hidden_sizes, 
                        hidden_acts, dropout_prob)

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('critic_output', nn.Linear(hidden_sizes[-1], 1))
        self.net[-1].apply(lambda m: init_weights(m, gain=1.0))

        # Create and initialize target Q network
        self.net_target = deepcopy(self.net)
        for param in self.net_target.parameters():
            param.requires_grad = False

class RegSACMLPActorCritic(SACMLPActorCritic):
    def __init__(self, env: VectorEnv, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic, dropout_prob):
        nn.Module.__init__(self)
        self.action_max = nn.Parameter(torch.tensor(env.single_action_space.high), 
                                       requires_grad=False)

        # Check the action space type and initialize the actor
        if isinstance(env.single_action_space, Box):
            obs_dim = env.single_observation_space.shape[0]
            act_dim = env.single_action_space.shape[0]
            self.actor = SACMLPActor(obs_dim, act_dim, hidden_sizes_actor, 
                                     hidden_acts_actor, self.action_max)
        else:
            raise NotImplementedError
        
        # Initialize the two critics
        self.critic_1 = RegSACMLPCritic(obs_dim, act_dim, hidden_sizes_critic,
                                        hidden_acts_critic, dropout_prob)
        self.critic_2 = RegSACMLPCritic(obs_dim, act_dim, hidden_sizes_critic,
                                        hidden_acts_critic, dropout_prob)
        
    def reset_weights(self):
        # Re-initialize actor
        self.actor.net[:-1].apply(lambda m: init_weights(m, gain=1.0))
        self.actor.net[-1].apply(lambda m: init_weights(m, gain=0.01))

        # Re-initialize the two critics
        for layer_1, layer_2 in zip(self.critic_1.net, self.critic_2.net):
            if isinstance(layer_1, nn.Linear):
                layer_1.apply(lambda m: init_weights(m, gain=1.0))
                layer_2.apply(lambda m: init_weights(m, gain=1.0))
            elif hasattr(layer_1, 'reset_parameters'):
                layer_1.reset_parameters()
                layer_2.reset_parameters()
        
        # Re-copy the weights of the critics into their target networks
        copy_parameters(self.critic_1.net.parameters(), self.critic_1.net_target.parameters())
        copy_parameters(self.critic_2.net.parameters(), self.critic_2.net_target.parameters())

class MLPDynamicsModel(MLP):
    def __init__(self, env: VectorEnv, hidden_sizes, hidden_acts):
        # Initialize MLP hidden layers
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        super().__init__('dynamics', obs_dim + act_dim, hidden_sizes, hidden_acts)

        # Add the output layer to the network and intialize its weights 
        self.net.add_module('dynamics_output', nn.Linear(hidden_sizes[-1], obs_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=1.0)) 
    
    def forward(self, obs, act):
        return self.net.forward(torch.cat([obs, act], dim=-1))
    
    def layer_summary(self):
        print("Dynamics Model Summary: \n")
        return super().layer_summary()

    def calc_pred_error(self, device, obs: np.ndarray, act: np.ndarray, obs_next: np.ndarray):
        with torch.no_grad():
            to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            obs, act, obs_next = to_tensor(obs), to_tensor(act), to_tensor(obs_next)

            obs_pred = self.forward(obs, act)
            pred_error = F.mse_loss(obs_next, obs_pred)
        
        return pred_error.cpu().numpy()

    def reset_weights(self):
        self.net.apply(lambda m: init_weights(m, gain=1.0))