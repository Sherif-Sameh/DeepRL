import os
import sys
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers.vector import RescaleAction
import numpy as np
import torch
import torch.nn.functional as F

from core.td3.train import TD3Trainer
from core.td3.train import ReplayBuffer as TD3ReplayBuffer
from core.td3.train import get_parser as get_parser_td3

from exploration.her.models.mlp import HER_TD3ActorCritic
from exploration.utils import is_multigoal_env
from exploration.rl_utils import MultiGoalObservation

class HERReplayBuffer:
    def __init__(self, env: AsyncVectorEnv, her_k, her_strat, buf_size):
        self.her_k = her_k
        self.env_buf_size = buf_size // env.num_envs
        
        # Ensure that HER sampling strategy is valid and set sampling method
        if her_strat in ['final', 'episode', 'future']:
            her_strats = {
                'final': self.sample_final,
                'episode': self.sample_episode,
                'future': self.sample_future
            }
            self.sample_her = her_strats[her_strat]
        else:
            raise ValueError('Invalid HER sampling strategy given')
        if her_strat == 'final': self.her_k = 1

        # Initialize buffers for storing episode start and end indices
        # for each transition to use during experience re-labeling
        self.ep_start_env = np.zeros(env.num_envs, dtype=np.int64)
        self.ep_start = np.zeros((env.num_envs, self.env_buf_size), dtype=np.int64)
        self.ep_end = np.zeros((env.num_envs, self.env_buf_size), dtype=np.int64)

        # Store a copy of the environment for reward and terminated calculations
        self.env_copy = env.unwrapped.env_fns[0]()
        
        # Store slices for retrieving individual parts of the observations
        self.observation = self.env_copy.obs_slice
        self.desired_goal = self.env_copy.dg_slice
        self.achieved_goal = self.env_copy.ag_slice
        self.obs_dg = self.env_copy.obs_dg_slice 

    def sample_final(self, env_indices, indices):
        # Get indices of final transition in each episode
        return self.ep_end[env_indices, indices] - 1

    def sample_episode(self, env_indices, indices):
        # Get indices from each episode 
        indices_episode = np.random.randint(
            low=self.ep_start[env_indices, indices],
            high=self.ep_end[env_indices, indices],
            size=len(indices)
            )
        
        return indices_episode
    
    def sample_future(self, env_indices, indices):
        # Get indices from each episode after the selected transition
        indices_future = np.random.randint(
            low=indices+1,
            high=self.ep_end[env_indices, indices],
            size=len(indices)
            )
        
        return indices_future

    def terminate_ep(self, env_id, ep_len):
        ep_len += 1 # Accounts for final observation stored at num transitions + 1

        # Update start and end indices for all transitions in the episode
        ep_end = min(self.ep_start_env[env_id] + ep_len, self.env_buf_size)
        self.ep_start[env_id, self.ep_start_env[env_id]:ep_end] = self.ep_start_env[env_id]
        self.ep_end[env_id, self.ep_start_env[env_id]:ep_end] = ep_end 

        # Update new start index and wrap around if needed
        if (self.ep_start_env[env_id] + ep_len) > self.env_buf_size:
            ep_len_remaining = (self.ep_start_env[env_id] + ep_len) - self.env_buf_size
            self.ep_start[env_id, :ep_len_remaining] = 0
            self.ep_end[env_id, :ep_len_remaining] = ep_len_remaining
            self.ep_start_env[env_id] = ep_len_remaining 
        else:
            self.ep_start_env[env_id] = ep_end

class HER_TD3ReplayBuffer(HERReplayBuffer, TD3ReplayBuffer):
    def __init__(self, env, her_k, her_strat, buf_size, batch_size):
        TD3ReplayBuffer.__init__(self, env, buf_size, batch_size)
        HERReplayBuffer.__init__(self, env, her_k, her_strat, buf_size)

    def update_buffer(self, env_id, obs, act, rew, done):
        # Mark transitions from incomplete episodes to mask them during updates
        self.ep_start[env_id, self.ctr[env_id]] = -1
        self.ep_end[env_id, self.ctr[env_id]] = self.env_buf_size
        super().update_buffer(env_id, obs, act, rew, done)
    
    def get_batch(self):
        get_indices = lambda num_envs, size: \
            (np.random.randint(0, num_envs, size=self.batch_size),
             np.random.choice(size-1, self.batch_size, replace=False)) 
        num_envs = self.obs.shape[0]
        full_bs = self.batch_size * (self.her_k + 1) 
        size = np.min(np.where(self.buf_full == True, self.env_buf_size, self.ctr))

        # Initialize empty batches for storing real and re-labeled samples
        obs = np.zeros((full_bs, self.obs_shape[0]), dtype=np.float32)
        act = np.zeros((full_bs, self.act_shape[0]), dtype=np.float32)
        rew = np.zeros(full_bs, dtype=np.float32)
        obs_next = np.zeros((full_bs, self.obs_shape[0]), dtype=np.float32)
        done = np.zeros(full_bs, dtype=np.bool)
        mask = np.zeros(full_bs, dtype=np.bool)

        # Sample real experiences
        env_indices, exp_indices = get_indices(num_envs, size)
        exp_indices = np.where(exp_indices==(self.ep_end[env_indices, exp_indices]-1),
                               exp_indices-1, exp_indices)
        indices_slice = slice(0, self.batch_size)
        obs[indices_slice] = self.obs[env_indices, exp_indices]
        act[indices_slice] = self.act[env_indices, exp_indices]
        rew[indices_slice] = self.rew[env_indices, exp_indices]
        obs_next[indices_slice] = self.obs[env_indices, exp_indices+1]
        done[indices_slice] = self.done[env_indices, exp_indices]
        mask[indices_slice] = True
        
        # Sample and re-label experiences
        for k in range(self.her_k): 
            # Sample experience indices
            env_indices, exp_indices = get_indices(num_envs, size)
            exp_indices = np.where(exp_indices==(self.ep_end[env_indices, exp_indices]-1),
                                   exp_indices-1, exp_indices)
            indices_slice = slice(self.batch_size * (k+1), self.batch_size * (k+2))
            
            # Sample HER goals
            exp_her_indices = self.sample_her(env_indices, exp_indices)
            her_goals = self.obs[env_indices, exp_her_indices, self.achieved_goal]

            # Re-calculate and assign re-labeled experiences
            obs[indices_slice, self.observation] = self.obs[env_indices, exp_indices, self.observation]
            obs[indices_slice, self.desired_goal] = her_goals
            act[indices_slice] = self.act[env_indices, exp_indices]
            rew[indices_slice] = self.env_copy.compute_reward(self.obs[env_indices, exp_indices+1, 
                                                                       self.achieved_goal], her_goals)
            obs_next[indices_slice, self.observation] = self.obs[env_indices, exp_indices+1, self.observation]
            obs_next[indices_slice, self.desired_goal] = her_goals
            done[indices_slice] = self.env_copy.compute_terminated(self.obs[env_indices, exp_indices+1, 
                                                                            self.achieved_goal], her_goals)
            mask[indices_slice] = self.ep_start[env_indices, exp_indices] > 0
        
        return obs, act, rew, obs_next, done, mask
    
    def terminate_ep(self, env_id, ep_len):
        # Update epsisode start and end indices
        TD3ReplayBuffer.terminate_ep(self, env_id)
        HERReplayBuffer.terminate_ep(self, env_id, ep_len)

        # Erase next episode that will be overriden by new transitions
        if self.buf_full[env_id] == True:
            start = self.ep_start_env[env_id]
            self.ep_start[env_id, start:self.ep_end[env_id, start]] = -1

class HER_TD3Trainer(TD3Trainer):
    """
    Hindsight Experience Replay (HER)

    :param her_k: Number of additional goals selected for each real experience tuple.
        Automatically set to 1 if the goal sampling strategy is 'final'.
    :param her_strat: Goal sampling strategy used for sampling additional goals. Can
        be any of 'final', 'episode' or 'future' as per the original HER paper. 
    :param buf_size: Remains size of the overall replay buffer used for storing real
        experiences. Additional re-labeled experiences are generated during sampling 
        and thus do not take up extra space in the buffer.
    :param batch_size: Number of real experiences sampled per environment. Therefore
        the total batch size = (batch_size * num_envs) * (her_k + 1)
    """
    def __init__(self, her_k, her_strat, env_fn, buf_size, batch_size, **td3_args):
        # Initialize actor-critic trainer
        super().__init__(env_fn=env_fn, buf_size=buf_size, batch_size=batch_size, **td3_args)

        # Replace normal replay buffer with HER replay buffer
        self.buf = HER_TD3ReplayBuffer(self.env, her_k, her_strat, buf_size, 
                                       batch_size * self.env.num_envs)
    
    def _calc_q_loss(self, obs, act, rew, obs_next, done, loss_mask):
        return super()._calc_q_loss(self.ac_mod.slice_observation(obs), act, rew,
                                    self.ac_mod.slice_observation(obs_next), done, loss_mask)
        
    def _calc_pi_loss(self, obs, loss_mask):
        return super()._calc_pi_loss(self.ac_mod.slice_observation(obs), loss_mask)
    
    def _proc_env_rollout(self, env_id, ep_len):
        self.buf.terminate_ep(env_id, ep_len)
        self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)
    
    def _end_training(self):
        super()._end_training()
        self.buf.env_copy.close()

def get_algo_type():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['td3'])

    # Parse first argument and pass on remaining ones
    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    return args.algo, remaining_args

if __name__ == '__main__':
    # Parse first argument to know algorithm type
    algo, remaining_args = get_algo_type()

    # Get parser for all remaining DRL algorithm arguments and add HER arguments 
    if algo == 'td3':
        parser = get_parser_td3()

    # HER Arguments
    parser.add_argument('--her_k', type=int, default=4)
    parser.add_argument('--her_strat', type=str, default='future')

    args = parser.parse_args(remaining_args)
    args.exp_name = 'her_'+ args.exp_name
    assert args.policy == 'mlp', "HER must be used with a MLP policy"

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Check that environment is multi-goal compatible
    is_multigoal_env(args.env)

    # Setup actor-critic kwargs
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if algo == 'td3':
        ac = HER_TD3ActorCritic
        ac_kwargs = dict(hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri,
                         hidden_acts_actor=torch.nn.ReLU, 
                         hidden_acts_critic=torch.nn.ReLU,
                         action_std=args.action_std,
                         action_std_f=args.action_std_f)
        env_fn_def = lambda render_mode=None: MultiGoalObservation(
            gym.make(args.env, max_episode_steps=max_ep_len, render_mode=render_mode))
        env_fn = [env_fn_def] * args.cpu
        wrappers_kwargs = dict()

    # Setup the trainer and begin training
    if algo == 'td3':
        trainer = HER_TD3Trainer(her_k=args.her_k, her_strat=args.her_strat, env_fn=env_fn, 
                                 wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                                 model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
                                 seed=args.seed, steps_per_epoch=args.steps, buf_size=args.buf_size, 
                                 gamma=args.gamma, polyak=args.polyak, lr=args.lr, lr_f=args.lr_f, 
                                 pre_act_coeff=args.pre_act_coeff, norm_rew=args.norm_rew, 
                                 norm_obs=args.norm_obs, max_grad_norm=args.max_grad_norm, 
                                 clip_grad=args.clip_grad, batch_size=args.batch_size, 
                                 start_steps=args.start_steps, learning_starts=args.learning_starts, 
                                 update_every=args.update_every, num_updates=args.num_updates, 
                                 target_noise=args.target_noise, noise_clip=args.noise_clip, 
                                 policy_delay=args.policy_delay, num_test_episodes=args.num_test_episodes, 
                                 seq_len=args.seq_len, seq_prefix=args.seq_prefix, 
                                 seq_stride=args.seq_stride, log_dir=log_dir, save_freq=args.save_freq, 
                                 checkpoint_freq=args.checkpoint_freq)
        
    trainer.learn(args.epochs, ep_init=args.ep_init)