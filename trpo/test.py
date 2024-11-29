import gym
import cv2
import numpy as np
import signal
import datetime
from pathlib import Path
from spinup.utils.test_policy import load_policy_and_env, run_policy

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, output_dir="./videos/", fps=30):
        super().__init__(env)
        self.frames = [] 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps  

    def render(self, mode='rgb_array', **kwargs):
        self.frames.append(super().render(mode, **kwargs))

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
    parser.add_argument('--data', type=str, default='../data/trpo/trpo_custom_discrete')
    parser.add_argument('--itr', type=int, default=-1)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--record', type=bool, default=False)
    parser.add_argument('--video_dir', type=str, default='../video/trpo/')
    parser.add_argument('--video_fps', type=int, default=30)
    args = parser.parse_args()

    itr = 'last' if args.itr == -1 else args.itr
    env, get_action = load_policy_and_env(args.data, itr=itr)
    if args.record == True:
        env_name = getattr(env.unwrapped.spec, "id", env.unwrapped.__class__.__name__)
        env = VideoRecorder(env, output_dir=args.video_dir + '/' + env_name + '/', fps=args.video_fps)

    def signal_handler(signum, frame):
        env.close()
        print('Simulation Terminated')
    signal.signal(signal.SIGINT, signal_handler)

    run_policy(env, get_action, num_episodes=args.num_episodes)
    env.close()