import numpy as np
import torch
from torch import nn

def init_weights(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        module.bias.data.fill_(0)

""" Inverse dynamics model that is responsible for estimating the action taken by the 
agent given the current and following features extracted from the raw observations. """
class InverseModel(nn.Module):
    def __init__(self, feature_dim, act_dim, hidden_sizes):
        super().__init__()
        self.feature_dim = feature_dim
        self.act_dim = act_dim

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]
        
        # Initialize hidden fc layers
        self.net = nn.Sequential()
        hidden_sizes = [2 * feature_dim] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            self.net.add_module(f'inv_hidden_{i+1}', nn.Linear(hidden_sizes[i], 
                                                               hidden_sizes[i+1]))
            self.net.add_module(f'inv_activation_{i+1}', nn.ReLU())
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Add output layer and initialize it
        self.net.add_module('inv_output', nn.Linear(hidden_sizes[-1], act_dim))
        self.net[-1].apply(lambda m: init_weights(m, gain=0.01))
    
    def forward(self, features):
        # features shape (batch_size, seq_len, feature_dim)
        features_curr = features[:, :-1, :].flatten(0, 1)
        features_next = features[:, 1:, :].flatten(0, 1)

        act_pred = self.net(torch.cat([features_curr, features_next], dim=-1))
        act_pred = act_pred.view(features.shape[0], features.shape[1]-1, self.act_dim)

        return act_pred
    
    def layer_summary(self):
        x = torch.randn((1, 2 * self.feature_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')

""" Forward dynamics model that is responsible for estimating the next feature vector given 
the current features and the actual action taken by the agent between the two states. """
class ForwardModel(nn.Module):
    def __init__(self, feature_dim, act_dim, hidden_sizes):
        super().__init__()
        self.feature_dim = feature_dim
        self.act_dim = act_dim

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]
        
        # Initialize hidden fc layers
        self.net = nn.Sequential()
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
        features_curr = features[:, :-1, :].flatten(0, 1)
        act_curr = act[:, :-1, :].flatten(0, 1)

        features_next_pred = self.net(torch.cat([features_curr, act_curr], dim=-1))
        features_next_pred = features_next_pred.view(features.shape[0], features.shape[1]-1, 
                                                     self.feature_dim)
        
        return features_next_pred
    
    def layer_summary(self):
        x = torch.randn((1, self.feature_dim + self.act_dim))
        for layer in self.net:
            input_shape = x.shape
            x = layer(x)
            print(layer.__class__.__name__, 'input & output shapes:\t', input_shape, x.shape)
        print('\n')