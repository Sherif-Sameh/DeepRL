import os
import pickle

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.vector import NormalizeReward as NormalizeRewardVector
from gymnasium.wrappers.vector import NormalizeObservation as NormalizeObservationVector


''' Performs the Polyak averaging operation used to update the parameters of a target network in
off-policy DRL algorithms like DDPG, TD3, and SAC. '''
def polyak_average(params, target_params, polyak):
    with torch.no_grad():
        for param, param_target in zip(params, target_params):
            param_target.data.mul_(polyak)
            param_target.data.add_(param.data, alpha=1-polyak)

''' Copies the values of parameters in the main network to the target network '''
def copy_parameters(params, target_params):
    with torch.no_grad():
        for param, param_target in zip(params, target_params):
            param_target.copy_(param)

''' Environment wrapper for skipping a certain number of observations using action-repeat 
and Scaling uint8 image observations from 0 to 255 to 32-bit floats from 0.0 to 1.0''' 
class SkipAndScaleObservation(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total_reward += rew
            if terminated or truncated:
                break

        return obs.astype(np.float32)/255.0, total_reward, terminated, truncated, info

''' Modified version of vector NormalizeObservation wrapper that does not automatically 
    normalize observations, but only updates their running mean and variance. '''
class NormalizeObservationManual(NormalizeObservationVector):
    def observations(self, observations):
        self.obs_rms.update(observations)
        return observations
    
    def normalize_observations(self, observations):
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )

''' Modified version of vector NormalizeReward wrapper that does not automatically 
    normalize rewards, but only updates their running mean and variance. '''
class NormalizeRewardManual(NormalizeRewardVector):
    def step(self, actions):
        obs, reward, terminated, truncated, info = super(NormalizeRewardVector, self).step(actions)
        self.accumulated_reward = (
            self.accumulated_reward * self.gamma * (1 - terminated) + reward
        )
        self.return_rms.update(self.accumulated_reward)
        return obs, reward, terminated, truncated, info

    def normalize_rewards(self, rewards):
        return rewards / np.sqrt(self.return_rms.var + self.epsilon)

''' Environment wrapper that extends NormalizeObservation to accept initial values 
and freezes running mean and variance updates. Used for evaluation. '''
class NormalizeObservationFrozen(NormalizeObservation):
    def __init__(self, env, mean=0, var=1.0, epsilon = 1e-8):
        super().__init__(env, epsilon)
        self.obs_rms.mean = mean
        self.obs_rms.var = var
        self.update_running_mean = False

''' Traverses the heirarchy of wrappers to find the given method, call it and return
all returned values from that method. '''
def call_env_method(env, method: str, *method_args):
    while env is not None:
        if hasattr(env, method):
            return getattr(env, method)(*method_args)
        env = getattr(env, 'env', None)
    
    raise AttributeError(f'Base environment nor any of its wrapped versions \
                         has the method {method}')

''' Traverses the hierarchy of wrappers to find the NormalizeObservation wrapper
and return its running mean and variance values '''
def get_obs_mean_var(env):
    while env is not None:
        if hasattr(env, 'obs_rms'):
            return np.copy(env.obs_rms.mean), np.copy(env.obs_rms.var)
        env = getattr(env, 'env', None) 
    
    return 0.0, 1.0

''' Save a Gymnasium environment using Pickle '''
def save_env(env_fn, wrappers_kwargs, save_dir, render_mode='human'):
    # Create temporary environment copy and add TimeLimit wrapper kwargs
    base_env = env_fn(render_mode=render_mode)
    wrappers_kwargs['TimeLimit'] = {'max_episode_steps': base_env.spec.max_episode_steps}
    
    # Unwrap and store all environment wrappers
    wrappers = []
    while hasattr(base_env, 'env'):
        wrappers.append(type(base_env))
        base_env = base_env.env
    
    # Create environment state dictionary
    state_dict = {
        'base_env_class': base_env.__class__,
        'base_env_kwargs': base_env.spec.kwargs,  
        'wrappers': wrappers,
        'wrappers_kwargs': wrappers_kwargs
    }

    with open(os.path.join(save_dir, f'env_{str(render_mode)}.pkl'), 'wb') as f:
        pickle.dump(state_dict, f)
    
    base_env.close()

''' Update the stored running mean and variance for normalize observation wrapper '''
def update_mean_var_env(env, save_dir, render_mode='human'):
    obs_mean, obs_var = get_obs_mean_var(env)
    filepath = os.path.join(save_dir, f'env_{str(render_mode)}.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        if 'NormalizeObservationFrozen' in state_dict['wrappers_kwargs']:
            state_dict['wrappers_kwargs']['NormalizeObservationFrozen'] = \
                {'mean': obs_mean, 'var': obs_var}
            with open(filepath, 'wb') as f:
                pickle.dump(state_dict, f)

''' Load a Gymnasium environment using Pickle '''
def load_env(save_dir, render_mode='human'):
    with open(os.path.join(save_dir, f'env_{str(render_mode)}.pkl'), 'rb') as f:
        state_dict = pickle.load(f)
    
    # Create base environment
    base_env_class = state_dict['base_env_class']
    base_env_kwargs = state_dict['base_env_kwargs']
    env = base_env_class(**base_env_kwargs)

    # Re-wrap environment using all its stored wrappers
    wrappers_kwargs = state_dict['wrappers_kwargs']
    for wrapper in reversed(state_dict['wrappers']):
        env = wrapper(env, **(wrappers_kwargs.get(wrapper.__name__, dict())))
    
    return env

''' Run a vectorized environment for the given number of episodes '''
def run_env(env: VectorEnv, num_episodes):
        # Run a few episodes to stabilize return variance estimate
        num_episodes_done = 0
        env.reset()
        while num_episodes_done < num_episodes * env.num_envs:
            act = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(act)
            done = np.logical_or(terminated, truncated)
            for env_id in range(env.num_envs):
                if done[env_id]: 
                    num_episodes_done += 1