import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers.vector import RescaleAction, NormalizeReward
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from core.sac.models.mlp import MLPActorCritic
from core.sac.models.cnn import CNNActorCritic
from core.rl_utils import SkipAndScaleObservation, save_env
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
        self.logp = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.rew = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.q_val = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.done = np.zeros((env.num_envs, self.env_buf_size), dtype=np.bool)

    def update_buffer(self, env_id, obs, act, logp, rew, q_val, done):
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
        self.logp[env_id, self.ctr[env_id]] = logp
        self.rew[env_id, self.ctr[env_id]] = rew
        self.q_val[env_id, self.ctr[env_id]] = q_val
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
        logp = np.zeros((self.batch_size,), dtype=np.float32)
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
            logp[env_slice] = self.logp[env_id, indices]
            rew[env_slice] = self.rew[env_id, indices]
            obs_next[env_slice] = self.obs[env_id, indices+1]
            done[env_slice] = self.done[env_id, indices]

        # Return randomly selected experience tuples
        return to_tensor(obs, torch.float32), to_tensor(act, torch.float32), \
                to_tensor(logp, torch.float32), to_tensor(rew, torch.float32), \
                to_tensor(obs_next, torch.float32), to_tensor(done, torch.bool)
    
    def get_buffer_size(self):
        return np.sum(self.env_buf_size * self.buf_full + self.ctr * (1 - self.buf_full))
    

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
                 max_grad_norm=0.5, clip_grad=False, alpha=0.2, alpha_min=3e-3, 
                 entropy_target=None, auto_alpha=True, batch_size=100, 
                 start_steps=10000, learning_starts=1000, update_every=50,
                 num_updates=-1, num_test_episodes=10, log_dir=None, 
                 save_freq=10, checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
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
        self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0) \
            if isinstance(self.env.single_action_space, Box) else self.env # Rescale cont. action spaces to [-1, 1]
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
        self.buf = ReplayBuffer(self.env, buf_size, batch_size * self.env.num_envs)

        # Initialize log alpha module based on training configuration
        if auto_alpha == True:
            alpha_init = alpha if alpha is not None else 1.0
            self.entropy_target = -np.prod(self.env.action_space.shape, dtype=np.float32) \
                if entropy_target is None else entropy_target
            self.alpha_mod = AlphaModule(alpha_init=alpha_init, requires_grad=True)
        else:
            self.entropy_target = None
            self.alpha_mod = AlphaModule(alpha_init=alpha, requires_grad=False)

        # Initialize optimizers
        self.ac_optim = Adam(self.ac_mod.parameters(), lr=lr)
        self.alpha_optim = Adam(self.alpha_mod.parameters(), lr=lr)

    def __calc_q_loss(self, alpha, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor, 
                      obs_next: torch.Tensor, done: torch.Tensor):
        # Get current Q values from both Q networks
        q_vals_1 = self.ac_mod.critic_1.forward(obs, act)
        q_vals_2 = self.ac_mod.critic_2.forward(obs, act)
        q_vals_1.squeeze_(dim=1)
        q_vals_2.squeeze_(dim=1)
        
        # Get actions and their log probs for next observations
        act_target, log_prob_target = self.ac_mod.actor.log_prob_no_grad(obs_next)

        # Determine target Q values and mask values where done is True
        q_vals_target_1 = self.ac_mod.critic_1.forward_target(obs_next, act_target)
        q_vals_target_2 = self.ac_mod.critic_2.forward_target(obs_next, act_target)
        q_vals_target = torch.min(q_vals_target_1, q_vals_target_2)
        q_vals_target.squeeze_(dim=1)
        q_vals_target -= alpha * log_prob_target
        q_vals_target[done] = 0.0

        # Calculate TD target and errors
        td_target = rew + self.gamma * q_vals_target

        return F.mse_loss(q_vals_1, td_target), F.mse_loss(q_vals_2, td_target)
    
    def __calc_pi_loss(self, alpha, obs: torch.Tensor):
        # Re-propagate actions and their log probs through actor for gradient tracking
        act, log_prob = self.ac_mod.actor.log_prob(obs)  

        # Evaluate the minimum of the two critics' Q values
        q_vals_1 = self.ac_mod.critic_1.forward_actions(obs, act)
        q_vals_2 = self.ac_mod.critic_2.forward_actions(obs, act)
        q_vals = torch.min(q_vals_1, q_vals_2)
        q_vals.squeeze_(dim=1)

        return (-q_vals + alpha * log_prob).mean()

    def __calc_alpha_loss(self, logp: torch.Tensor):
        alpha = self.alpha_mod.forward()
        
        return (alpha * (-logp - self.entropy_target)).mean()
    
    def __update_params(self, epoch):
        # Get mini-batch from replay buffer
        obs, act, logp, rew, obs_next, done = self.buf.get_batch(self.device)
        
        alpha_det = self.alpha_mod.forward().detach().to(self.device)
        
        # Get critics loss
        self.ac_optim.zero_grad()
        loss_q1, loss_q2 = self.__calc_q_loss(alpha_det, obs, act, 
                                              rew, obs_next, done)

        # Get actor loss (critic's weights are frozen temporarily)
        self.ac_mod.critic_1.set_grad_tracking(val=False)
        self.ac_mod.critic_2.set_grad_tracking(val=False)
        loss_pi = self.__calc_pi_loss(alpha_det, obs)
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
            loss_alpha = self.__calc_alpha_loss(logp.cpu())
            loss_alpha.backward()
            self.alpha_optim.step()
            with torch.no_grad(): 
                self.alpha_mod.log_alpha.clamp_min_(torch.tensor(np.log(self.alpha_min)))
            self.writer.add_scalar('Loss/LossAlpha', loss_alpha.item(), epoch+1)

        # Update target critic networks
        self.ac_mod.critic_1.update_target(self.polyak)
        self.ac_mod.critic_2.update_target(self.polyak)
        
        # Log training statistics
        self.writer.add_scalar('Loss/LossQ1', loss_q1.item(), epoch+1)
        self.writer.add_scalar('Loss/LossQ2', loss_q2.item(), epoch+1)
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)

    def train_mod(self, epochs=100):
        # Initialize scheduler
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        ac_scheduler = LinearLR(self.ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)
        alpha_scheduler = LinearLR(self.alpha_optim, start_factor=1.0, end_factor=end_factor,
                                   total_iters=epochs)

        # Normalize returns for more stable training
        if not isinstance(self.env, NormalizeReward):
            self.env = NormalizeReward(self.env)
        
        # Initialize environment variables
        obs, _ = self.env.reset(seed=self.seed)
        start_time = time.time()
        autoreset = np.zeros(self.env.num_envs)
        q_vals = []

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                if (step + self.steps_per_epoch*epoch) > self.start_steps:
                    act, q_val, logp = self.ac_mod.step(torch.as_tensor(obs, dtype=torch.float32, 
                                                                        device=self.device))
                else:
                    act = self.env.action_space.sample()
                    _, q_val, logp = self.ac_mod.step(torch.as_tensor(obs, dtype=torch.float32, 
                                                                      device=self.device))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], logp[env_id], 
                                               rew[env_id], q_val[env_id], terminated[env_id])
                        q_vals.append(q_val[env_id])
                obs = obs_next
                autoreset = np.logical_or(terminated, truncated)

                if (self.buf.get_buffer_size() >= self.learning_starts) \
                    and ((step % self.update_every) == 0):
                    for _ in range(self.num_updates):
                        self.__update_params(epoch)
            
            # Evaluate deterministic policy (skip return normalization wrapper)
            env = self.env.env if isinstance(self.env, NormalizeReward) else self.env 
            ep_len, ep_ret = np.zeros(env.num_envs), np.zeros(env.num_envs)
            ep_lens, ep_rets = [], []
            obs, _ = env.reset()
            while len(ep_lens) < self.num_test_episodes*env.num_envs:
                act = self.ac_mod.act(torch.as_tensor(obs, dtype=torch.float32, 
                                                      device=self.device))
                obs, rew, terminated, truncated, _ = env.step(act)
                ep_len, ep_ret = ep_len + 1, ep_ret + rew
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    for env_id in range(env.num_envs):
                        if done[env_id]:
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret[env_id])
                            ep_len[env_id], ep_ret[env_id] = 0, 0
            obs, _ = self.env.reset()
            ac_scheduler.step()
            alpha_scheduler.step()

            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))
            
            # Log info about epoch
            total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
            ep_lens, ep_rets, q_vals = np.array(ep_lens), np.array(ep_rets), np.array(q_vals)
            self.writer.add_scalar('EpLen/mean', ep_lens.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/mean', ep_rets.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/max', ep_rets.max(), total_steps_so_far)
            self.writer.add_scalar('EpRet/min', ep_rets.min(), total_steps_so_far)
            self.writer.add_scalar('QVals/mean', q_vals.mean(), epoch+1)
            self.writer.add_scalar('QVals/max', q_vals.max(), epoch+1)
            self.writer.add_scalar('QVals/min', q_vals.min(), epoch+1)
            self.writer.add_scalar('Alpha', self.alpha_mod.forward().item(), epoch+1)
            self.writer.add_scalar('Time', time.time()-start_time, epoch+1)
            self.writer.flush()
            q_vals = []
        
        # Save final model
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        self.env.close()
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')

    # Common model arguments 
    parser.add_argument('--hid_act', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--hid_cri', nargs='+', type=int, default=[64, 64])

    # CNN model arguments
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', nargs='+', type=int, default=[32, 64, 64])
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[8, 4, 3])
    parser.add_argument('--strides', nargs='+', type=int, default=[4, 2, 1])
    parser.add_argument('--features_out', nargs='+', type=int, default=[512])

    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--alpha_min', type=float, default=3e-3)
    parser.add_argument('--auto_alpha', type=bool, default=True)
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
    elif args.policy == 'cnn':
        ac = CNNActorCritic
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides, 
                         features_out=args.features_out,
                         hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri)
        env_fn_def = lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                       render_mode=render_mode)
        env_fn = [lambda render_mode=None: FrameStackObservation(SkipAndScaleObservation(
            GrayscaleObservation(env_fn_def(render_mode=render_mode))), stack_size=args.in_channels)] * args.cpu
        wrappers_kwargs = {
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
                         alpha=args.alpha, alpha_min=args.alpha_min, auto_alpha=args.auto_alpha, 
                         batch_size=args.batch_size, start_steps=args.start_steps, 
                         learning_starts=args.learning_starts, update_every=args.update_every, 
                         num_updates=args.num_updates, num_test_episodes=args.num_test_episodes, 
                         log_dir=log_dir, save_freq=args.save_freq, 
                         checkpoint_freq=args.checkpoint_freq)

    trainer.train_mod(args.epochs)