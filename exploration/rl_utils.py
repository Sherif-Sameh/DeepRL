import cv2
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium import Wrapper, ObservationWrapper

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
    
''' Wrapper implementing the reward function used for training the Unitree-Go1 
    Quadruped robot in the paper for Adaptive Policy ReguLarization (APRL) available
    here: https://sites.google.com/berkeley.edu/aprl. Does not reward smooth torque 
    transitions as the actions are not fed back as observations in this setup. '''
class APRLQuadrupedWrapper(Wrapper):
    def __init__(self, 
                 env: gym.Env,
                 Kp: float = 100,
                 Kd: float = 0.0,
                 home_q: Optional[np.ndarray] = None,
                 fv_tar: float = 2.0,
                 yr_tar: float = 0.0,
                 vel_coeff: float = 2.0,
                 q_coeff: float = 2.0,
                 rr_coeff: float = 0.02,
                 pr_coeff: float = 0.04,
                 yr_coeff: float = 0.4,
                 healthy_coeff: float = 1.0,
                 energy_coeff: float = 2e-4):
        super().__init__(env)
        self.q_pos_prev = np.zeros(12)

        # Store PD controller gains for torque estimation
        self.Kp = Kp
        self.Kd = Kd

        # Store home postion and target velocities (forward and yaw)
        self.home_q = np.deg2rad([0, 45, -90] * 4) if home_q is None else home_q
        self.env.unwrapped.init_qpos[-17:] = np.array([0.325, 1, 0, 0, 0, *self.home_q])
        self.joint_weights = np.array([1.0, 0.75, 0.5] * 4)
        self.fv_tar = fv_tar
        self.yr_tar = yr_tar

        # Modify action space to be zero centered and offset by the home position
        self.act_high = self.action_space.high - self.home_q
        self.act_low = self.action_space.low - self.home_q  
        self.action_space = spaces.Box(-1, 1, (12,), dtype=self.action_space.dtype)

        # Store reward coefficients 
        self.vel_coeff = vel_coeff
        self.q_coeff = q_coeff
        self.rr_coeff = rr_coeff
        self.pr_coeff = pr_coeff
        self.yr_coeff = yr_coeff
        self.healthy_coeff = healthy_coeff
        self.energy_coeff = energy_coeff

        # Define slices to extract individual parts from observation
        self.quat = slice(1, 5)
        self.q_pos = slice(5, 17)
        self.lin_vel = slice(17, 20)
        self.ang_vel = slice(20, 23)
        self.q_vel = slice(23, 35)

    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.q_pos_prev = np.copy(obs[self.q_pos])

        return obs, info
    
    def step(self, action):
        q_pos_tar = np.where(action < 0, -action * self.act_low + self.home_q, 
                             action * self.act_high + self.home_q)
        obs, _, terminated, truncated, info = self.env.step(q_pos_tar)
        reward = self._compute_reward(obs, q_pos_tar)

        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, observation, q_pos_tar):
        # Get all the individual parts of the observation separately
        roll, pitch, _, q_pos, lin_vel, ang_vel, q_vel = self.__slice_observation(observation)

        # Forward velocity component
        fwd_vel = np.cos(pitch) * lin_vel[0]
        up_cos_dist = np.cos(roll) * np.cos(pitch)
        c_upright = (0.5 * up_cos_dist + 0.5)**2
        fv_rew = np.maximum(np.abs(self.fv_tar), 0.2) \
            * self.__nqr(fwd_vel, self.fv_tar, self.fv_tar, 1.6) * c_upright

        # Joint pose component
        q_pos_err = self.joint_weights * (q_pos - self.home_q)
        jp_rew = np.exp(-0.6 * np.sum(q_pos_err**2))

        # Roll, Pitch, and Yaw rate components
        rr_rew = np.abs(ang_vel[0])**1.4
        pr_rew = np.abs(ang_vel[1])**1.4
        yr_rew = self.__nqr(ang_vel[2], self.yr_tar, np.maximum(self.yr_tar, np.pi/6), 1.6)

        # Healthy component (penalize if unhealthy)
        healthy_rew = (np.abs(roll) > np.pi/4) or (np.abs(pitch) > np.pi/4)

        # Energy component
        torque = self.__torque(q_pos_tar, self.q_pos_prev, q_vel)
        energy_rew = np.sum(np.abs(q_vel * torque))
        self.q_pos_prev = q_pos

        # Return combined reward
        return self.vel_coeff * fv_rew \
            + self.q_coeff * jp_rew \
            + self.yr_coeff * yr_rew \
            - self.rr_coeff * rr_rew \
            - self.pr_coeff * pr_rew \
            - self.healthy_coeff * healthy_rew \
            - self.energy_coeff * energy_rew

    def __slice_observation(self, observation):
        # Get pose and joint angles
        roll, pitch, yaw = self.__quat_to_euler(*observation[self.quat])
        q_pos = observation[self.q_pos]

        # Get body and joint velocities
        lin_vel, ang_vel = observation[self.lin_vel], observation[self.ang_vel]
        q_vel = observation[self.q_vel]

        return roll, pitch, yaw, q_pos, lin_vel, ang_vel, q_vel
    
    def __quat_to_euler(self, w, i, j, k):
        roll = np.atan2(2 * (w * i + j * k), 1 - 2 * (i**2 + j**2))
        pitch = np.asin(2 * (w * j - k * i))
        yaw = np.atan2(2 * (w * k + i * j), 1 - 2 * (j**2 + k**2))

        return roll, pitch, yaw
    
    def __nqr(self, v, t, b, p):
        delta = np.abs((v - t)/b)
        
        return (1 - delta**p) if delta <= 1 else (1 - delta)  

    def __torque(self, q_tar, q_pos, q_vel):
        return self.Kp * (q_tar - q_pos) - self.Kd * q_vel