import sys
sys.path.append('../../')
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from core.ppo.models.mlp import MLPActorCritic

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
    
env_fn = lambda: gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, 
                       domain_randomize=False, continuous=False)
env_fn_wrapped = lambda: FrameStackObservation(SkipAndScaleObservation(GrayscaleObservation(env_fn())), stack_size=4)
env = gym.vector.AsyncVectorEnv([env_fn_wrapped] * 4)
ac = MLPActorCritic(env, [128, 128], [128, 128], torch.nn.ReLU, torch.nn.ReLU)

if __name__ == '__main__':
    for episode in range(10):
        env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            a = env.action_space.sample()
            s, r, terminated, truncated, info = env.step(a)
            done = np.any(np.logical_or(terminated, truncated))
        print(s.dtype, s[0, 0, :15])
        print(f'Episode {episode+1} done \n')
    env.close()