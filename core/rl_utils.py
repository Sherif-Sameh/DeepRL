import os
import pickle

import numpy as np
import torch
import gymnasium as gym


''' Performs the Polyak averaging operation used to update the parameters of a target network in
off-policy DRL algorithms like DDPG, TD3, and SAC. '''
def polyak_average(params, target_params, polyak):
    with torch.no_grad():
        for param, param_target in zip(params, target_params):
            param_target.data.mul_(polyak)
            param_target.data.add_(param.data, alpha=1-polyak)

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
        'base_env': base_env,
        'wrappers': wrappers,
        'wrappers_kwargs': wrappers_kwargs
    }

    with open(os.path.join(save_dir, f'env_{str(render_mode)}.pkl'), 'wb') as f:
        pickle.dump(state_dict, f)
    
    base_env.close()

''' Load a Gymnasium environment using Pickle '''
def load_env(save_dir, render_mode='human'):
    with open(os.path.join(save_dir, f'env_{str(render_mode)}.pkl'), 'rb') as f:
        state_dict = pickle.load(f)
    
    # Create base environment
    env = state_dict['base_env']

    # Re-wrap environment using all its stored wrappers
    wrappers_kwargs = state_dict['wrappers_kwargs']
    for wrapper in reversed(state_dict['wrappers']):
        env = wrapper(env, **(wrappers_kwargs.get(wrapper.__name__, dict())))
    
    return env