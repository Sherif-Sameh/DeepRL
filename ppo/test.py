import os
import datetime
import cv2
from pathlib import Path
import gymnasium as gym
import torch

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, output_dir="./videos/", fps=30):
        super().__init__(env)
        self.frames = [] 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        video_path = self.output_dir / f'{str(datetime.datetime.now())}.mp4'

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
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--data', type=str, default='../runs/ppo')
    parser.add_argument('--itr', type=int, default=-1)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--record', type=bool, default=False)
    parser.add_argument('--video_dir', type=str, default='../video/ppo/')
    parser.add_argument('--video_fps', type=int, default=30)
    args = parser.parse_args()

    model_name = 'model.pt' if args.itr < 0 else f'model{args.itr}.pt'
    model_path = os.path.join(args.data, f'pyt_save/{model_name}')
    model = torch.load(model_path, weights_only=False)
    assert hasattr(model, 'act'), 'model must be implement an act(obs) method getting actions'

    render_mode = 'rgb_array' if args.record==True else 'human'
    env = gym.make(args.env, render_mode=render_mode)
    env.metadata['render_fps'] = args.video_fps
    if args.record == True:
        env_name = getattr(env.unwrapped.spec, "id", env.unwrapped.__class__.__name__)
        video_dir = os.path.join(args.video_dir, f'{env_name}')
        env = VideoRecorder(env, video_dir, fps=args.video_fps)

    for ep in range(args.num_episodes):
        (obs, _), done = env.reset(), False
        ep_ret, ep_len = 0, 0
        while not done:
            if args.record == True:
                env.render()
            act = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_ret, ep_len = ep_ret + rew, ep_len + 1
            done = terminated or truncated
        print(f'Episode {ep}: EpRet = {ep_ret} \t EpLen = {ep_len}')

    env.close()