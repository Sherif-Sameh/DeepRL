import cv2
import numpy as np
from gymnasium import spaces
from gymnasium import ObservationWrapper
from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward

''' Observation wrapper for ViZDoom environments to extract normal RGB image 
    observations from Dict observation and resize to a given image size'''
class VizdoomToGymnasium(ObservationWrapper):
    def __init__(self, env, img_size=84):
        super().__init__(env)
        self.img_size = img_size   

        num_channels = self.observation_space['screen'].shape[-1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(img_size, img_size, num_channels),
            dtype=np.uint8,
        )

    def observation(self, observation):
        # Extract RGB from observation dict and resize to given image size
        return cv2.resize(observation['screen'], 
                          (self.img_size, self.img_size), 
                          interpolation=cv2.INTER_AREA)

''' Observation wrapper used for Gymnasium-Robotics environments to flatten their 
    Dict observation as well as provide convenient wrappers for the compute reward, 
    truncation and termination methods without requring an info Dict. '''
class MultiGoalObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        goal_dim = env.observation_space['desired_goal'].shape[0]
        observation_dim = env.observation_space['observation'].shape[0]
        
        # Define slices for extracting the 3 different parts of an observation
        self.obs_slice = slice(0, observation_dim)
        self.dg_slice = slice(observation_dim, observation_dim + goal_dim)
        self.ag_slice = slice(observation_dim + goal_dim, observation_dim + 2 * goal_dim)
        self.obs_dg_slice = slice(0, observation_dim + goal_dim)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim + 2 * goal_dim,),
            dtype=np.float64
        )
    
    def reset(self, *, seed = None, options = None):
        state_is_goal = True
        while state_is_goal:
            obs, info = super().reset(seed=seed, options=options)
            state_is_goal = (self.compute_reward(obs[self.ag_slice], obs[self.dg_slice]) == 0)
        
        return obs, info
    
    def observation(self, observation):
        # Flatten multi-goal dict observation in the order
        # (observation, desired_goal, achieved_goal)
        return np.concat([observation['observation'], 
                          observation['desired_goal'],
                          observation['achieved_goal']])
    
    # Wrap compute reward, termination and truncation methods to pass
    # an empty 'info' dict as 'info' is not stored in any of the algorithms 
    def compute_reward(self, achieved_goal, desired_goal):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, dict())
    
    def compute_terminated(self, achieved_goal, desired_goal):
        return self.env.unwrapped.compute_terminated(achieved_goal, desired_goal, dict())
    
    def compute_truncated(self, achieved_goal, desired_goal):
        return self.env.unwrapped.compute_truncated(achieved_goal, desired_goal, dict())