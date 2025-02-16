import torch
import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from exploration.aprl.models.mlp import MLPDynamicsModel

class RegularizeAction(VectorWrapper):
    def __init__(self, env: VectorEnv, device, growth_rate, shrink_rate, 
                 act_start, act_end, dyn_threshold, dyn_mod: MLPDynamicsModel):
        super().__init__(env)
        self.device = device
        self.step_ctr = 0
        self.growth_rate = growth_rate
        self.shrink_rate = shrink_rate
        self.act_curr = act_start
        self.act_init = act_start
        self.act_end = act_end
        self.dyn_threshold = dyn_threshold
        self.dyn_mod = dyn_mod
        
        # Store additional environment variables
        self.active = True
        self.obs_prev = np.zeros(env.observation_space.shape)
        self.valid_transition = np.ones(self.env.num_envs)
        def identity(x): return x
        self.obs_transform = self.env.normalize_observations if hasattr(self.env, "normalize_observations") else identity

    def reset(self, *, seed = None, options = None):
        observations, infos = super().reset(seed=seed, options=options)

        # Reset wrappers internal states
        if self.active == True:
            self.obs_prev = self.obs_transform(np.copy(observations))
            self.valid_transition = np.ones(self.env.num_envs)

        return observations, infos
    
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        if self.active == True:
            # Update action regularization limit based on given parameters
            obs_next = self.obs_transform(np.copy(observations))
            self.step_ctr += self.env.num_envs
            alpha_curr = np.clip(self.step_ctr/self.growth_rate, 0, 1)
            self.act_curr = alpha_curr * self.act_end + (1-alpha_curr) * self.act_init

            # Calculate dynamics error and update action limits if required
            pred_error = self.dyn_mod.calc_pred_error(self.device, self.obs_prev, actions, obs_next)
            pred_error *= (self.valid_transition.sum()/self.env.num_envs)
            if pred_error >= self.dyn_threshold:
                self.step_ctr = 0
                self.act_init = self.shrink_rate * self.act_curr
        
            # Update internal states for next iteration
            self.obs_prev = obs_next
            self.valid_transition = np.logical_not(np.logical_or(terminations, truncations))

        return observations, rewards, terminations, truncations, infos
    
    def get_step_count(self):
        return self.step_ctr

    def get_action_limit(self):
        return self.act_curr

    def enable_act_reg(self):
        self.active = True

    def disable_act_reg(self):
        self.active = False