import sys
sys.path.append('../../')
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from core.rl_utils import SkipAndScaleObservation, save_env, load_env

env_fn = lambda: gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, 
                       domain_randomize=False, continuous=False)
env_fn_wrapped = lambda: FrameStackObservation(SkipAndScaleObservation(GrayscaleObservation(env_fn())), stack_size=4)
env = env_fn_wrapped()
wrappers_kwargs = {
    'TimeLimit': {'max_episode_steps': env.spec.max_episode_steps},
    'FrameStackObservation': {'stack_size': 4}
}
try:
    save_env(env, wrappers_kwargs, save_dir=os.getcwd())
except Exception as e:
    print(f'Failed to save environment: {e}')

# try:
#     env = load_env(os.getcwd())
# except Exception as e:
#     print(f'Failed to load environment: {e}')

if __name__ == '__main__':
    # exit()
    for episode in range(10):
        env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            a = env.action_space.sample()
            s, r, terminated, truncated, info = env.step(a)
            done = np.any(np.logical_or(terminated, truncated))
        print(f'Episode {episode+1} done \n')
    env.close()