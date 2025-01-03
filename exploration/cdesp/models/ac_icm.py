import torch
import torch.nn.functional as F 
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box, Discrete

from exploration.cdesp.models.icm import IntrinsicCuriosityModule
from core.ppo.models.cnn_lstm import CNNLSTMActorCritic as PPOActorCritic
from core.td3.models.cnn_lstm import CNNLSTMActorCritic as TD3ActorCritic

# Wrapper over ICM module for PPO Actor Critic
class PPOICM(IntrinsicCuriosityModule):
    def __init__(self, env: VectorEnv, ac_mod: PPOActorCritic, 
                 eta, hidden_size_inv, hidden_sizes_fwd):
        self.ac_mod = ac_mod
        
        # Extract needed parameters from environment and actor-critic
        if isinstance(env.single_action_space, Discrete):
            act_dim = env.single_action_space.n
            self.act_transform = lambda act: F.one_hot(act, num_classes=act_dim)
        elif isinstance(env.single_action_space, Box):
            act_dim = env.single_action_space.shape[0]
            self.act_transform = lambda act: act
        feature_dim = ac_mod.feature_ext.features_out

        # Initialize ICM
        super().__init__(eta, feature_dim, act_dim, hidden_size_inv, hidden_sizes_fwd)
        
    def extract_features(self, obs):
        return self.ac_mod.feature_ext(obs.unsqueeze(1))
    
    def extract_features_no_grad(self, obs):
        with torch.no_grad(): features = self.ac_mod.feature_ext(obs.unsqueeze(1))

        return features
    
    def step(self, obs, features):
        # Evaluate act, val and logp without re-propagating through feature extractor
        act = self.ac_mod.actor.forward(obs, features=features)
        val = self.ac_mod.critic.forward(obs, features=features)
        logp = self.ac_mod.actor.log_prob_no_grad(act)

        return act.squeeze(1).cpu().numpy(), val.squeeze(1).cpu().numpy(), \
            logp.squeeze(1).cpu().numpy()
    
    def calc_reward(self, features, features_next, act):
        act = self.act_transform(act.unsqueeze(1))
        intrinsic_rew = super().calc_reward(features, features_next, act)

        return intrinsic_rew.squeeze(1).cpu().numpy()

# Wrapper over ICM module for TD3 Actor Critic
class TD3ICM(IntrinsicCuriosityModule):
    def __init__(self, env: VectorEnv, ac_mod: TD3ActorCritic, 
                 eta, hidden_size_inv, hidden_sizes_fwd):
        self.ac_mod = ac_mod
        
        # Extract needed parameters from environment and actor-critic
        act_dim = env.single_action_space.shape[0]
        feature_dim = ac_mod.feature_ext.features_out

        # Initialize ICM
        super().__init__(eta, feature_dim, act_dim, hidden_size_inv, hidden_sizes_fwd)
        
    def extract_features(self, obs):
        return self.ac_mod.feature_ext(obs.unsqueeze(1))
    
    def extract_features_no_grad(self, obs):
        with torch.no_grad(): features = self.ac_mod.feature_ext(obs.unsqueeze(1))

        return features
    
    def step(self, obs, features):
        # Evaluate act and q_val without re-propagating through feature extractor
        with torch.no_grad():
            act = self.ac_mod.actor.forward(obs, features=features)
            act = torch.max(torch.min(act + self.ac_mod.action_std * torch.randn_like(act),
                                      self.ac_mod.action_max), -self.ac_mod.action_max)
            q_val = self.ac_mod.critic_1.forward(obs, act, features=features)

            return act.squeeze(1).cpu().numpy(), q_val.cpu().numpy().squeeze()
    
    def calc_reward(self, features, features_next, act):
        act = act.unsqueeze(1)
        intrinsic_rew = super().calc_reward(features, features_next, act)

        return intrinsic_rew.squeeze(1).cpu().numpy()