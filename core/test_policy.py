import os
import datetime
import cv2
from pathlib import Path
from gymnasium.core import Env

import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from core.rl_utils import load_env

# Import all polcies (will look to do this in a cleaner manner in the future)

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, output_dir="./videos/", exp_name='', fps=30):
        super().__init__(env)
        self.frames = [] 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self.fps = fps  

    def render(self):
        self.frames.append(super().render())

    def close(self):
        self.save_video()
        super().close()

    def save_video(self):
        if len(self.frames) == 0:
            return

        # Define video file path
        video_path = self.output_dir / f'{self.exp_name}_{str(datetime.datetime.now())}.mp4'

        # Get the height and width from the first frame
        height, width, _ = self.frames[0].shape

        # Define the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

        # Write each frame to the video
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        writer.release()
        print(f"Saved video: {video_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--itr', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--record', type=bool, default=False)
    parser.add_argument('--video_dir', type=str, default='../video/')
    parser.add_argument('--video_fps', type=int, default=30)
    args = parser.parse_args()
    
    # Load the saved model
    model_name = 'model.pt' if args.itr < 0 else f'model{args.itr}.pt'
    model_path = os.path.join(args.run, f'pyt_save/{model_name}')
    model = torch.load(model_path, weights_only=False) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.use_gpu else torch.device('cpu')
    model.to(device)
    assert hasattr(model, 'act'), 'model must be implement an act(obs) method getting actions'

    # Attempt to load saved environment or initialize new one if loading fails
    render_mode = 'rgb_array' if args.record==True else 'human'
    try:
        env = load_env(save_dir=args.run, render_mode=render_mode)
    except Exception as e:
        print(f'Could not load saved environment: {e} \n\n')
        print(f'Initializing new environment using given arguments')
        env = gym.make(args.env, render_mode=render_mode)

    if isinstance(env.action_space, gym.spaces.Box): 
        env = RescaleAction(env, min_action=-1.0, max_action=1.0) 
    _ = env.reset(seed=args.seed)

    # Initialize video recorder if needed
    if args.record == True:
        exp_name = os.path.basename(args.run[:-1])
        env_name = getattr(env.unwrapped.spec, "id", env.unwrapped.__class__.__name__)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        video_dir = os.path.join(current_script_dir, args.video_dir, f'{env_name}')
        env = VideoRecorder(env, video_dir, exp_name=exp_name, fps=args.video_fps)

    # Run policy in environment for the set number of episodes
    for ep in range(args.num_episodes):
        (obs, _), done = env.reset(), False
        ep_ret, ep_len = 0, 0
        while not done:
            if args.record == True:
                env.render()
            act = model.act(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_ret, ep_len = ep_ret + rew, ep_len + 1
            done = terminated or truncated
        print(f'Episode {ep}: EpRet = {ep_ret} \t EpLen = {ep_len}')

    env.close()