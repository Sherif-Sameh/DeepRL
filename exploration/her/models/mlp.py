### Only used to wrap step, act, and forward methods in actor-critics to remove 
### 'achieved_goal' from passed observations for use with HER

import torch
from gymnasium.vector import VectorEnv
from core.td3.models.mlp import MLPActorCritic as TD3ActorCritic
from core.td3.models.mlp import MLPActor as TD3Actor
from core.td3.models.mlp import MLPCritic as TD3Critic

class HER_TD3ActorCritic(TD3ActorCritic):
    def __init__(self, env: VectorEnv, hidden_sizes_actor, hidden_sizes_critic,
                 hidden_acts_actor, hidden_acts_critic, action_std, action_std_f):
        super().__init__(env, hidden_sizes_actor, hidden_sizes_critic, hidden_acts_actor, 
                         hidden_acts_critic, action_std, action_std_f)
        self.obs_dg_slice = env.unwrapped.get_attr('obs_dg_slice')[0]

        # Re-initialize actor and critics with correct observation dimension
        obs_dim = self.obs_dg_slice.stop - self.obs_dg_slice.start
        self.act_dim = env.single_action_space.shape[0]
        self.actor = TD3Actor(obs_dim, self.act_dim, hidden_sizes_actor, 
                              hidden_acts_actor, self.action_max)
        self.critic_1 = TD3Critic(obs_dim, self.act_dim, hidden_sizes_critic,
                                  hidden_acts_critic)
        self.critic_2 = TD3Critic(obs_dim, self.act_dim, hidden_sizes_critic,
                                  hidden_acts_critic)

    def slice_observation(self, obs):
        return obs[:, self.obs_dg_slice]
    
    def step(self, obs):
        return super().step(self.slice_observation(obs))
    
    def act(self, obs, deterministic=False):
        if obs.ndim == 1:
            obs = self.slice_observation(obs.unsqueeze(0)).squeeze(0)
        else:
            obs = self.slice_observation(obs)
        
        return super().act(obs, deterministic=deterministic)
    
    def forward(self, obs):
        return super().forward(self.slice_observation(obs))