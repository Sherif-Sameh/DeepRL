import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from exploration.cdesp.models.icm import IntrinsicCuriosityModule

class IntrinsicRewardWrapper(VectorWrapper):
    def __init__(self, rew_icm_max, env: VectorEnv, icm_mod: IntrinsicCuriosityModule, device):
        super().__init__(env)
        self.rew_icm_max = rew_icm_max
        self.icm_mod = icm_mod
        self.device = device

        self.active = True
        self.obs_prev = np.zeros(env.observation_space.shape)
        self.valid_transition = np.zeros(self.env.num_envs)
        self.returns_intrinsic = np.zeros(self.env.num_envs)
        self.returns_extrinsic = np.zeros(self.env.num_envs)

    def reset(self, *, seed = None, options = None):
        observations, infos = super().reset(seed=seed, options=options)
        
        # Reset internal states
        if self.active == True:
            self.obs_prev = np.copy(observations)
            self.returns_intrinsic = np.zeros(self.env.num_envs)
            self.returns_extrinsic = np.zeros(self.env.num_envs)
            self.valid_transition = np.ones(self.env.num_envs)

        return observations, infos
    
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        # Calculate intrinsic rewards
        if self.active == True:
            rewards_intrinsic = self.icm_mod.calc_reward(self.obs_prev, observations, actions, self.device)
            rewards_intrinsic = self.valid_transition * rewards_intrinsic
            rewards_intrinsic = np.minimum(rewards_intrinsic, self.rew_icm_max)

            # Update internal states for next step() call
            self.obs_prev = np.copy(observations)
            self.valid_transition = np.logical_not(np.logical_or(terminations, truncations))
            
            # Update returns and return combined reward
            self.returns_intrinsic += rewards_intrinsic
            self.returns_extrinsic += rewards
        else:
            rewards_intrinsic = np.zeros_like(rewards)
            
        return observations, rewards+rewards_intrinsic, terminations, truncations, infos
    
    def get_and_clear_return(self, env_idx):
        ep_return_intrinsic = self.returns_intrinsic[env_idx]
        ep_return_extrinsic = self.returns_extrinsic[env_idx]
        self.returns_intrinsic[env_idx] = 0
        self.returns_extrinsic[env_idx] = 0

        return ep_return_intrinsic, ep_return_extrinsic

    def disable_intrinsic_reward(self):
        self.active = False

    def enable_intrinsic_reward(self):
        self.active = True