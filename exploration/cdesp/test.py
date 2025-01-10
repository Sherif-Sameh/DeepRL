import os
import cv2
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, ClipAction
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers import  AddRenderObservation, ResizeObservation
from core.rl_utils import SkipAndScaleObservation

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--itr', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--deterministic', action="store_true", default=False)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--record', action="store_true", default=False)
    parser.add_argument('--video_dir', type=str, default='../video/')
    parser.add_argument('--video_fps', type=int, default=30)
    args = parser.parse_args()

    # Seed NumPy, PyTorch and GPU
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu == True:
        torch.cuda.manual_seed(seed=args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    env = FrameStackObservation(
        SkipAndScaleObservation(
        GrayscaleObservation(
        ResizeObservation(
        AddRenderObservation(gym.make('MountainCar-v0', max_episode_steps=500, render_mode="rgb_array")), shape=(42, 42))), skip=2), stack_size=3)
    
    if isinstance(env.action_space, gym.spaces.Box): 
        env = ClipAction(RescaleAction(env, min_action=-1.0, max_action=1.0)) 
    env.reset(seed=args.seed)
    
    # Load the saved model
    model_name = 'model.pt' if args.itr < 0 else f'model{args.itr}.pt'
    model_path = os.path.join(args.run, f'pyt_save/{model_name}')
    model = torch.load(model_path, weights_only=False) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.use_gpu else torch.device('cpu')
    model.to(device)
    assert hasattr(model, 'act'), 'model must be implement an act(obs) method getting actions'

    # Run policy in environment for the set number of episodes
    for ep in range(10):
        (obs, _), done = env.reset(), False
        ep_ret, ep_len = 0, 0
        print(obs.shape)
        if hasattr(model, 'reset_hidden_states'):
            model.reset_hidden_states(device)
        while not done:
            act = model.act(torch.as_tensor(obs, dtype=torch.float32).to(device),
                            deterministic=args.deterministic)
            obs, rew, terminated, truncated, _ = env.step(act)
            rgb_array = env.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imshow('Environment', bgr_array)
            cv2.waitKey(1000//60)
            ep_ret, ep_len = ep_ret + rew, ep_len + 1
            done = terminated or truncated
        print(f'Episode {ep}: EpRet = {ep_ret} \t EpLen = {ep_len}')

    env.close()