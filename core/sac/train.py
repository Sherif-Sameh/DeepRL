import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers.vector import RescaleAction, NormalizeReward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from core.sac.models.mlp import MLPActorCritic
from core.sac.models.cnn import CNNActorCritic
from core.sac.models.cnn_lstm import CNNLSTMActorCritic
from core.rl_utils import SkipAndScaleObservation, save_env, run_env
from core.utils import serialize_locals, clear_logs

class ReplayBuffer:
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size):
        # Check the type of the action space
        if not (isinstance(env.single_action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.ctr, self.buf_full = np.zeros(env.num_envs, dtype=np.int64), np.full(env.num_envs, False)
        self.env_buf_size, self.batch_size = buf_size // env.num_envs, batch_size
        self.obs_shape = env.single_observation_space.shape
        self.act_shape = env.single_action_space.shape

        # Initialize all buffers for storing data during training
        self.obs = np.zeros((env.num_envs, self.env_buf_size) + self.obs_shape, dtype=np.float32)
        self.act = np.zeros((env.num_envs, self.env_buf_size) + self.act_shape, dtype=np.float32)
        self.rew = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.done = np.zeros((env.num_envs, self.env_buf_size), dtype=np.bool)

    def update_buffer(self, env_id, obs, act, rew, done):
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
        self.rew[env_id, self.ctr[env_id]] = rew
        self.done[env_id, self.ctr[env_id]] = done

        # Update buffer counter and reset if neccessary
        self.ctr[env_id] += 1
        if self.ctr[env_id] == self.env_buf_size:
            self.ctr[env_id] = 0
            self.buf_full[env_id] = True

    def get_batch(self, device):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype,
                                                          device=device)
        env_bs = self.batch_size // self.obs.shape[0]

        # Initialize empty batches for storing samples from the environments
        obs = np.zeros((self.batch_size,)+self.obs_shape, dtype=np.float32)
        act = np.zeros((self.batch_size,)+self.act_shape, dtype=np.float32)
        rew = np.zeros(self.batch_size, dtype=np.float32)
        obs_next = np.zeros((self.batch_size,)+self.obs_shape, dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.bool)

        # Generate random indices and sample experience tuples
        for env_id in range(self.obs.shape[0]):
            size = self.env_buf_size if self.buf_full[env_id]==True else self.ctr[env_id]
            indices = np.random.choice(size-1, env_bs, replace=False)
            env_slice = slice(env_bs * env_id, env_bs * (env_id+1))

            obs[env_slice] = self.obs[env_id, indices]
            act[env_slice] = self.act[env_id, indices]
            rew[env_slice] = self.rew[env_id, indices]
            obs_next[env_slice] = self.obs[env_id, indices+1]
            done[env_slice] = self.done[env_id, indices]

        # Return randomly selected experience tuples
        return to_tensor(obs, torch.float32), to_tensor(act, torch.float32), \
                to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                to_tensor(done, torch.bool)
    
    def get_buffer_size(self):
        return np.mean(np.where(self.buf_full, self.env_buf_size, self.ctr))
    
    def increment_ep_num(self, env_id):
        pass

class SequenceReplayBuffer(ReplayBuffer):
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size, 
                 seq_len, seq_prefix, stride):
        super().__init__(env, buf_size, batch_size)
        self.seq_len = seq_len
        self.seq_prefix = seq_prefix
        self.stride = stride

        # Initialize array for storing the episode number of each experience tuple collected
        self.ep_nums = np.zeros((env.num_envs, self.env_buf_size), dtype=np.int64)
        self.ep_num_ctrs = np.zeros(env.num_envs, dtype=np.int64)

    def update_buffer(self, env_id, obs, act, rew, done):
        self.ep_nums[env_id, self.ctr[env_id]] = self.ep_num_ctrs[env_id]
        super().update_buffer(env_id, obs, act, rew, done)

        # Increment episode counter if buffer wrapped around
        if (self.ctr[env_id] == 0) and (self.buf_full[env_id]):
            self.ep_num_ctrs[env_id] += 1

    def get_batch(self, device):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype, 
                                                          device=device)
        num_envs = self.obs.shape[0]
        env_bs = self.batch_size // num_envs

        # Calculate number for sequences available for each environment
        num_experiences = np.where(self.buf_full, self.env_buf_size, self.ctr)
        num_sequences = (num_experiences - self.seq_len - 1) // self.stride + 1

        # Generate random sequence indices for all environments
        env_indices = []
        seq_indices = []
        for env_id in range(num_envs):
            indices = np.random.choice(num_sequences[env_id], env_bs, replace=False)
            env_indices.extend([env_id] * env_bs)
            seq_indices.extend(indices)
        env_indices = np.array(env_indices)
        start_indices = np.array(seq_indices) * self.stride

        # Create sequence indices arrays using arange
        seq_offsets = np.arange(self.seq_len)
        obs_indices = start_indices[:, None] + seq_offsets
        obs_next_indices = obs_indices + 1

        # Sample sequences from replay buffer
        obs = self.obs[env_indices[:, None], obs_indices]
        act = self.act[env_indices[:, None], obs_indices]
        rew = self.rew[env_indices[:, None], obs_indices]
        obs_next = self.obs[env_indices[:, None], obs_next_indices]
        done = self.done[env_indices[:, None], obs_indices]

        # Create sequence mask for prefix and multi-episode sequences
        ep_nums = self.ep_nums[env_indices[:, None], obs_indices]
        seq_mask = (ep_nums == ep_nums[:, :1])
        seq_mask[:, :self.seq_prefix] = False

        # Return randomly selected experience tuples
        return to_tensor(obs, torch.float32), to_tensor(act, torch.float32), \
                to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                to_tensor(done, torch.bool), to_tensor(seq_mask, torch.bool)
    
    def increment_ep_num(self, env_id):
        self.ep_num_ctrs[env_id] += 1

class AlphaModule(nn.Module):
    def __init__(self, alpha_init=1.0, requires_grad=True):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([alpha_init], dtype=torch.float32)), 
                                      requires_grad=requires_grad)

    def forward(self):
        return torch.exp(self.log_alpha)
    

class SACTrainer:
    def __init__(self, env_fn, wrappers_kwargs=dict(), use_gpu=False, model_path='', 
                 ac=MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=1000, 
                 buf_size=1000000, gamma=0.99, polyak=0.995, lr=3e-4, lr_f=None, 
                 max_grad_norm=0.5, clip_grad=False, alpha=0.2, alpha_min=0, 
                 entropy_target=None, auto_alpha=True, batch_size=100, 
                 start_steps=2500, learning_starts=1000, update_every=50,
                 num_updates=-1, num_test_episodes=10, seq_len=32, 
                 seq_prefix=16, seq_stride=10, log_dir=None, save_freq=10, 
                 checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = None
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.lr_f = lr_f
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad
        self.alpha_min = alpha_min
        self.auto_alpha = auto_alpha
        self.start_steps = start_steps
        self.learning_starts = learning_starts
        self.update_every = update_every
        self.num_updates = num_updates if num_updates > 0 else update_every
        self.num_test_episodes = num_test_episodes
        self.save_freq = save_freq
        self.checkpoint_freq = checkpoint_freq
        self.ep_len_list, self.ep_ret_list = [0], [0]
        
        # Serialize local hyperparameters
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn'); locals_dict.pop('wrappers_kwargs')
        locals_dict = serialize_locals(locals_dict)

        # Remove existing logs if run already exists
        clear_logs(log_dir)
        
        # Initialize logger and save hyperparameters
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(self.writer.get_logdir())}')
        self.save_dir = os.path.join(self.writer.get_logdir(), 'pyt_save')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize environment and attempt to save a copy of it 
        self.env = AsyncVectorEnv(env_fn)
        self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0) # Rescale cont. action spaces to [-1, 1]
        try:
            save_env(env_fn[0], wrappers_kwargs, log_dir, render_mode='human')
            save_env(env_fn[0], wrappers_kwargs, log_dir, render_mode='rgb_array')
        except Exception as e:
            print(f'Could not save environment: {e} \n\n')

        # Initialize Actor-Critic
        if len(model_path) > 0:
            self.ac_mod = torch.load(model_path, weights_only=False)
            self.ac_mod = self.ac_mod.to(torch.device('cpu'))
        else:
            self.ac_mod = ac(self.env, **ac_kwargs)
        self.ac_mod.layer_summary()
        self.writer.add_graph(self.ac_mod, torch.randn(size=self.env.observation_space.shape, dtype=torch.float32))

        # Setup random seed number for PyTorch and NumPy
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # GPU setup if necessary
        if use_gpu == True:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.manual_seed(seed=seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.ac_mod.to(self.device)

        # Initialize the experience replay buffer for training
        if ac != CNNLSTMActorCritic:
            self.buf = ReplayBuffer(self.env, buf_size, batch_size * self.env.num_envs)
        else:
            self.buf = SequenceReplayBuffer(self.env, buf_size, batch_size * self.env.num_envs,
                                            seq_len + seq_prefix, seq_prefix, seq_stride)
            
        # Initialize log alpha module based on training configuration
        if auto_alpha == True:
            alpha_init = alpha if alpha is not None else 1.0
            self.entropy_target = -np.prod(self.env.single_action_space.shape, dtype=np.float32) \
                if entropy_target is None else entropy_target
            self.alpha_mod = AlphaModule(alpha_init=alpha_init, requires_grad=True)
        else:
            self.entropy_target = None
            self.alpha_mod = AlphaModule(alpha_init=alpha, requires_grad=False)

        # Initialize optimizers
        self.ac_optim = Adam(self.ac_mod.parameters(), lr=lr)
        self.alpha_optim = Adam(self.alpha_mod.parameters(), lr=lr)

    def _calc_q_loss(self, alpha, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor, 
                      obs_next: torch.Tensor, done: torch.Tensor, loss_mask: torch.Tensor):
        # Determine target actions and their log probs for TD targets
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        act_target, log_prob_target = self.ac_mod.actor.log_prob_no_grad(obs_next)

        # Get current and target Q values from first critic
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_1 = self.ac_mod.critic_1.forward(obs, act)
        q_vals_target_1 = self.ac_mod.critic_1.forward_target(obs_next, act_target)
        q_vals_1.squeeze_(dim=-1)

        # Get current and target Q values from second critic
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_2 = self.ac_mod.critic_2.forward(obs, act)
        q_vals_target_2 = self.ac_mod.critic_2.forward_target(obs_next, act_target)
        q_vals_2.squeeze_(dim=-1)
        
        # Determine target Q values and mask values where done is True
        q_vals_target = torch.min(q_vals_target_1, q_vals_target_2)
        q_vals_target.squeeze_(dim=-1)
        q_vals_target -= alpha * log_prob_target
        q_vals_target[done] = 0.0

        # Calculate TD target and errors
        td_target = rew + self.gamma * q_vals_target
        td_target = td_target[loss_mask]

        return F.mse_loss(q_vals_1[loss_mask], td_target), F.mse_loss(q_vals_2[loss_mask], td_target)
    
    def _calc_pi_loss(self, alpha, obs: torch.Tensor, loss_mask: torch.Tensor):
        # Re-propagate actions and their log probs through actor for gradient tracking
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        act, log_prob = self.ac_mod.actor.log_prob(obs)  

        # Evaluate the minimum of the two critics' Q values
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_1 = self.ac_mod.critic_1.forward_actions(obs, act)
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_2 = self.ac_mod.critic_2.forward_actions(obs, act)
        q_vals = torch.min(q_vals_1, q_vals_2)
        q_vals.squeeze_(dim=-1)

        return (-q_vals[loss_mask] + alpha * log_prob[loss_mask]).mean(), log_prob

    def _calc_alpha_loss(self, logp: torch.Tensor, loss_mask: torch.Tensor):
        log_alpha = self.alpha_mod.log_alpha
        
        return -(log_alpha * (logp[loss_mask].detach().cpu() + self.entropy_target)).mean()
    
    def _update_params(self, epoch):
        # Get mini-batch from replay buffer
        batch = self.buf.get_batch(self.device)

        # Unpack experience tuple
        if len(batch) == 6:
            obs, act, rew, obs_next, done, mask = batch
        else:
            obs, act, rew, obs_next, done = batch
            mask = torch.full((obs.shape[0],), fill_value=True)
        
        alpha_det = self.alpha_mod.forward().detach().to(self.device)
        
        # Get critics loss
        self.ac_optim.zero_grad()
        loss_q1, loss_q2 = self._calc_q_loss(alpha_det, obs, act, rew, 
                                              obs_next, done, mask)

        # Get actor loss (critic's weights are frozen temporarily)
        self.ac_mod.critic_1.set_grad_tracking(val=False)
        self.ac_mod.critic_2.set_grad_tracking(val=False)
        loss_pi, logp = self._calc_pi_loss(alpha_det, obs, mask)
        self.ac_mod.critic_1.set_grad_tracking(val=True)
        self.ac_mod.critic_2.set_grad_tracking(val=True)
        
        # Combine losses and calculate gradients
        loss = loss_pi + 0.5 * (loss_q1 + loss_q2)
        loss.backward()

        # Clip gradients (if neccessary and update parameters)
        if self.clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.ac_mod.actor.parameters(), self.max_grad_norm)
        self.ac_optim.step()
        
        # Update the log alpha parameter if its estimated online 
        if self.auto_alpha == True:
            self.alpha_optim.zero_grad()
            loss_alpha = self._calc_alpha_loss(logp, mask)
            loss_alpha.backward()
            self.alpha_optim.step()
            if self.alpha_min > 0:    
                with torch.no_grad(): self.alpha_mod.log_alpha.clamp_min_(torch.tensor(np.log(self.alpha_min)))
            self.writer.add_scalar('Loss/LossAlpha', loss_alpha.item(), epoch+1)

        # Update target critic networks
        self.ac_mod.critic_1.update_target(self.polyak)
        self.ac_mod.critic_2.update_target(self.polyak)
        
        # Log training statistics
        self.writer.add_scalar('Loss/LossQ1', loss_q1.item(), epoch+1)
        self.writer.add_scalar('Loss/LossQ2', loss_q2.item(), epoch+1)
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
    
    def _proc_env_rollout(self, env_id, ep_len):
        self.buf.increment_ep_num(env_id)
        self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id) 
        ep_len[env_id] = 0
    
    def _train(self, epoch):
        self.ac_mod.reset_hidden_states(self.device, save=True)
        for _ in range(self.num_updates):
            self._update_params(epoch)
        self.ac_mod.reset_hidden_states(self.device, restore=True)

    def _eval(self):
        # Evaluate deterministic policy (skip return normalization wrapper)
        env = self.env.env if isinstance(self.env, NormalizeReward) else self.env 
        ep_len, ep_ret = np.zeros(env.num_envs), np.zeros(env.num_envs)
        self.ep_len_list, self.ep_ret_list = [], []
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
        obs, _ = env.reset()
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
        while len(self.ep_len_list) < self.num_test_episodes*env.num_envs:
            act = self.ac_mod.act(to_tensor(obs), deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_len, ep_ret = ep_len + 1, ep_ret + rew
            done = np.logical_or(terminated, truncated)
            for env_id in range(env.num_envs):
                if done[env_id]:
                    self.ep_len_list.append(ep_len[env_id])
                    self.ep_ret_list.append(ep_ret[env_id])
                    ep_len[env_id], ep_ret[env_id] = 0, 0
                    self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
    
    def _log_ep_stats(self, epoch, q_val_list):
        total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
        ep_len_np, ep_ret_np, q_val_np = np.array(self.ep_len_list), np.array(self.ep_ret_list), np.array(q_val_list)
        self.writer.add_scalar('EpLen/mean', ep_len_np.mean(), total_steps_so_far)
        self.writer.add_scalar('EpRet/mean', ep_ret_np.mean(), total_steps_so_far)
        self.writer.add_scalar('EpRet/max', ep_ret_np.max(), total_steps_so_far)
        self.writer.add_scalar('EpRet/min', ep_ret_np.min(), total_steps_so_far)
        self.writer.add_scalar('QVals/mean', q_val_np.mean(), epoch+1)
        self.writer.add_scalar('QVals/max', q_val_np.max(), epoch+1)
        self.writer.add_scalar('QVals/min', q_val_np.min(), epoch+1)
        self.writer.add_scalar('Alpha', self.alpha_mod.forward().item(), epoch+1)

    def learn(self, epochs=100, ep_init=10):
        # Initialize scheduler
        self.epochs = epochs
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        ac_scheduler = LinearLR(self.ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)
        alpha_scheduler = LinearLR(self.alpha_optim, start_factor=1.0, end_factor=end_factor,
                                   total_iters=epochs)
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Normalize returns for more stable training
        if not isinstance(self.env, NormalizeReward):
            self.env = NormalizeReward(self.env, gamma=self.gamma)
        self.env.reset(seed=self.seed)
        run_env(self.env, num_episodes=ep_init)

        # Initialize environment variables
        obs, _ = self.env.reset()
        ep_len = np.zeros(self.env.num_envs, dtype=np.int64)
        autoreset = np.zeros(self.env.num_envs)
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
        q_val_list = []

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                if (step + self.steps_per_epoch*epoch) > self.start_steps:
                    act, q_val = self.ac_mod.step(to_tensor(obs))
                else:
                    act = self.env.action_space.sample()
                    _, q_val = self.ac_mod.step(to_tensor(obs))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], 
                                               rew[env_id], terminated[env_id])
                        q_val_list.append(q_val[env_id])
                    else:
                        self._proc_env_rollout(env_id, ep_len)
                obs, ep_len = obs_next, ep_len + 1
                autoreset = np.logical_or(terminated, truncated)

                if (self.buf.get_buffer_size() >= self.learning_starts) \
                    and ((step % self.update_every) == 0):
                    self._train(epoch)
            
            self._eval()
            obs, _ = self.env.reset()
            ac_scheduler.step()
            alpha_scheduler.step()

            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))
            
            # Log info about epoch
            self._log_ep_stats(epoch, q_val_list)
            q_val_list = []
            self.writer.flush()
        
        # Save final model
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        self.env.close()
        print(f'Model {epochs} (final) saved successfully')

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument('--model_path', type=str, default='')

    # Common model arguments 
    parser.add_argument('--hid_act', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--hid_cri', nargs='+', type=int, default=[64, 64])

    # CNN model arguments (shared by all CNN policies)
    parser.add_argument('--action_rep', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', nargs='+', type=int, default=[32, 64, 64])
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[8, 4, 3])
    parser.add_argument('--strides', nargs='+', type=int, default=[4, 2, 1])
    parser.add_argument('--padding', nargs='+', type=int, default=[0, 0, 0])
    parser.add_argument('--features_out', nargs='+', type=int, default=[512])

    # CNN-LSTM specific model arguments
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--seq_prefix', type=int, default=16)
    parser.add_argument('--seq_stride', type=int, default=10)

    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ep_init', type=int, default=10)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--alpha_min', type=float, default=0)
    parser.add_argument('--entropy_target', type=float, default=None)
    parser.add_argument('--auto_alpha', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_steps', type=int, default=2500)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--num_updates', type=int, default=-1)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--cpu', type=int, default=4)
    
    return parser

if __name__ == '__main__':
    # Parse input arguments
    parser = get_parser()
    args = parser.parse_args()

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Determine type of policy and setup its arguments and environment
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if args.policy == 'mlp':
        ac = MLPActorCritic
        ac_kwargs = dict(hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri,
                         hidden_acts_actor=torch.nn.ReLU, 
                         hidden_acts_critic=torch.nn.ReLU)
        env_fn = [lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                    render_mode=render_mode)] * args.cpu
        wrappers_kwargs = dict()
    elif args.policy == 'cnn' or args.policy == 'cnn-lstm':
        ac = CNNActorCritic if args.policy == 'cnn' else CNNLSTMActorCritic 
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides, 
                         padding=args.padding,
                         features_out=args.features_out,
                         hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri)
        env_fn_def = lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                       render_mode=render_mode)
        env_fn = [lambda render_mode=None: FrameStackObservation(SkipAndScaleObservation(
            GrayscaleObservation(env_fn_def(render_mode=render_mode)), skip=args.action_rep), 
            stack_size=args.in_channels)] * args.cpu
        wrappers_kwargs = {
            'SkipAndScaleObservation': {'skip': args.action_rep},
            'FrameStackObservation': {'stack_size': args.in_channels}
        }
    else:
        raise NotImplementedError

    # Begin training
    trainer = SACTrainer(env_fn, wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                         model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
                         seed=args.seed, steps_per_epoch=args.steps, buf_size=args.buf_size, 
                         gamma=args.gamma, polyak=args.polyak, lr=args.lr, lr_f=args.lr_f, 
                         max_grad_norm=args.max_grad_norm, clip_grad=args.clip_grad, 
                         alpha=args.alpha, alpha_min=args.alpha_min, 
                         entropy_target=args.entropy_target, auto_alpha=args.auto_alpha, 
                         batch_size=args.batch_size, start_steps=args.start_steps, 
                         learning_starts=args.learning_starts, update_every=args.update_every, 
                         num_updates=args.num_updates, num_test_episodes=args.num_test_episodes, 
                         seq_len=args.seq_len, seq_prefix=args.seq_prefix, 
                         seq_stride=args.seq_stride, log_dir=log_dir, save_freq=args.save_freq, 
                         checkpoint_freq=args.checkpoint_freq)

    trainer.learn(epochs=args.epochs, ep_init=args.ep_init)