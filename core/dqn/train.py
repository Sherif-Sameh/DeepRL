import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers.vector import NormalizeReward
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from core.dqn.models.mlp import MLPDQN, MLPDuelingDQN, MLPDDQN, MLPDuelingDDQN
from core.dqn.models.cnn import CNNDQN, CNNDuelingDQN, CNNDDQN, CNNDuelingDDQN
from core.rl_utils import SkipAndScaleObservation, save_env
from core.utils import serialize_locals, clear_logs

class ReplayBuffer:
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size):
        # Check the type of the action space
        if not (isinstance(env.single_action_space, Discrete)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.ctr, self.buf_full = np.zeros(env.num_envs, dtype=np.int64), np.full(env.num_envs, False)
        self.env_buf_size, self.batch_size = buf_size // env.num_envs, batch_size
        self.obs_shape = env.single_observation_space.shape
        self.act_shape = env.single_action_space.shape

        # Initialize all buffers for storing data during training
        self.obs = np.zeros((env.num_envs, self.env_buf_size) + self.obs_shape, dtype=np.float32)
        self.act = np.zeros((env.num_envs, self.env_buf_size) + self.act_shape, dtype=np.int64)
        self.rew = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.q_val = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.done = np.zeros((env.num_envs, self.env_buf_size), dtype=np.bool)

    def update_buffer(self, env_id, obs, act, rew, q_val, done):
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
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
        act = np.zeros((self.batch_size,)+self.act_shape, dtype=np.int64)
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
        return to_tensor(obs, torch.float32), to_tensor(act, torch.int64), \
                to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                to_tensor(done, torch.bool)
    
    def get_buffer_size(self):
        return np.sum(self.env_buf_size * self.buf_full + self.ctr * (1 - self.buf_full))

    def get_weights(self, device):
        return torch.ones(self.batch_size, dtype=torch.float32, device=device)
    
    def update_beta(self):
        return
    
    def update_priorities(self, td_errors):
        return

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size, alpha, 
                 beta0, beta_rate, epsilon):
        super().__init__(env, buf_size, batch_size)

        # Store prioritized replay buffer parameters
        self.alpha = alpha
        self.beta = beta0
        self.beta_rate = beta_rate
        self.epsilon = epsilon

        # Initialize transition priority buffer and maximum priority value
        self.pri = np.zeros((env.num_envs, self.env_buf_size), dtype=np.float32)
        self.pri_max = np.ones(env.num_envs, dtype=np.float32)

        # Store weigths and indices for each sample in a mini-batch
        self.weights = np.ones((self.batch_size,), dtype=np.float32)
        self.indices = np.zeros((self.batch_size,), dtype=np.int64)

    def update_buffer(self, env_id, obs, act, rew, q_val, done):
        # Normal replay buffer updates
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
        self.rew[env_id, self.ctr[env_id]] = rew
        self.q_val[env_id, self.ctr[env_id]] = q_val
        self.done[env_id, self.ctr[env_id]] = done

        # Set priorities of new transitions to max priority
        self.pri[env_id, self.ctr[env_id]] = self.pri_max[env_id]

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
        
        for env_id in range(self.obs.shape[0]):
            # Calculate sampling proabilities for each transition
            size = self.env_buf_size if self.buf_full[env_id]==True else self.ctr[env_id]
            prob = np.zeros((size-1,), dtype=np.float32)
            prob = self.pri[env_id, :size-1]**self.alpha
            prob = prob / np.sum(prob)
        
            # Sample weighted random indices and update weights for parameter updates
            env_slice = slice(env_bs * env_id, env_bs * (env_id+1))
            self.indices[env_slice] = np.random.choice(size-1, env_bs, replace=False, p=prob)
            self.weights[env_slice] = (self.env_buf_size * prob[self.indices[env_slice]])**(-self.beta)
            self.weights[env_slice] = self.weights[env_slice] / np.max(self.weights[env_slice])

            obs[env_slice] = self.obs[env_id, self.indices[env_slice]]
            act[env_slice] = self.act[env_id, self.indices[env_slice]]
            rew[env_slice] = self.rew[env_id, self.indices[env_slice]]
            obs_next[env_slice] = self.obs[env_id, self.indices[env_slice]+1]
            done[env_slice] = self.done[env_id, self.indices[env_slice]]

        # Return randomly selected experience tuples
        return to_tensor(obs, torch.float32), to_tensor(act, torch.int64), \
                to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                to_tensor(done, torch.bool)

    def get_weights(self, device):
        return torch.as_tensor(self.weights, dtype=torch.float32, device=device)

    def update_beta(self):
        self.beta = min(self.beta + self.beta_rate, 1.0)

    def update_priorities(self, td_errors):
        env_bs = self.batch_size // self.obs.shape[0]
        for env_id in range(self.obs.shape[0]):
            env_slice = slice(env_bs * env_id, env_bs * (env_id+1))
            self.pri[env_id, self.indices[env_slice]] = np.abs(td_errors[env_slice]) + self.epsilon
            self.pri_max[env_id] = max(self.pri_max[env_id], np.max(self.pri[env_id, self.indices[env_slice]]))

    
class DQNTrainer:
    def __init__(self, policy, env_fn, wrappers_kwargs=dict(), use_gpu=False, model_path='', 
                 dueling=False, double_q=False, q_net_kwargs=dict(), seed=0, 
                 prioritized_replay=False, prioritized_replay_alpha=0.6, 
                 prioritized_replay_beta0=0.4, prioritized_replay_beta_rate=0.0, 
                 prioritized_replay_eps=1e-6, eps_init=1.0, eps_final=0.05, eps_decay_rate=0.2, 
                 buf_size=1000000, steps_per_epoch=1000, batch_size=100,
                 learning_starts=1000, train_freq=4, target_network_update_freq=500, 
                 num_test_episodes=10, gamma=0.99, lr=5e-4, lr_f=None, max_grad_norm=0.5, 
                 clip_grad=True, log_dir=None, save_freq=10, checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_network_update_freq = target_network_update_freq
        self.num_test_episodes = num_test_episodes
        self.gamma = gamma
        self.lr = lr
        self.lr_f = lr_f
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad
        self.save_freq = save_freq
        self.checkpoint_freq = checkpoint_freq

        # Serialize local hyperparameters
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn'); locals_dict.pop('wrappers_kwargs')
        locals_dict = serialize_locals(locals_dict)

        # Initialize logger and save hyperparameters
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(self.writer.get_logdir())}')
        self.save_dir = os.path.join(self.writer.get_logdir(), 'pyt_save')
        os.makedirs(self.save_dir, exist_ok=True)

        # Determine DQN variant to use
        if (dueling==True) and (double_q==True):
            q_net = MLPDuelingDDQN if policy == 'mlp' else CNNDuelingDDQN
        elif dueling==True:
            q_net = MLPDuelingDQN if policy == 'mlp' else CNNDuelingDQN
        elif double_q==True:
            q_net = MLPDDQN if policy == 'mlp' else CNNDDQN
        else:
            q_net = MLPDQN if policy == 'mlp' else CNNDQN

        # Initialize environment and attempt to save a copy of it 
        self.env = AsyncVectorEnv(env_fn)
        try:
            save_env(env_fn[0], wrappers_kwargs, self.writer.get_logdir(), render_mode='human')
            save_env(env_fn[0], wrappers_kwargs, self.writer.get_logdir(), render_mode='rgb_array')
        except Exception as e:
            print(f'Could not save environment: {e} \n\n')
        
        # Initialize Q-network
        if len(model_path) > 0:
            self.q_net_mod = torch.load(model_path, weights_only=False)
            self.q_net_mod = self.q_net_mod.to(torch.device('cpu'))
        else:
            self.q_net_mod = q_net(self.env, eps_init, eps_final, eps_decay_rate, **q_net_kwargs)
        self.q_net_mod.layer_summary()
        self.writer.add_graph(self.q_net_mod, torch.randn(size=self.env.observation_space.shape))

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
        self.q_net_mod.to(self.device)

        # Initialize the experience replay buffer
        if prioritized_replay==True:
            self.buf = PrioritizedReplayBuffer(self.env, buf_size, batch_size * self.env.num_envs, 
                                          prioritized_replay_alpha, prioritized_replay_beta0,
                                          prioritized_replay_beta_rate, prioritized_replay_eps)
        else:
            self.buf = ReplayBuffer(self.env, buf_size, batch_size * self.env.num_envs)

        # Initialize optimizer
        self.q_optim = Adam(self.q_net_mod.parameters(), lr=lr)
        
    def __calc_td_error(self, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor, 
                        obs_next: torch.Tensor, done: torch.Tensor):
        # Get current and target Q vals and mask target Q vals where done is True
        q_vals = self.q_net_mod.forward_grad(obs, act)
        q_vals_target = self.q_net_mod.forward_target(obs_next)
        q_vals_target[done] = 0.0

        # Calculate TD target and error
        td_target = rew + self.gamma * q_vals_target
        td_error = td_target - q_vals

        return td_error
    
    def __update_params(self, epoch, update_target_network):
        # Get mini-batch and perform one parameter update for the Q network
        obs, act, rew, obs_next, done = self.buf.get_batch(self.device)

        # Calculate Q-function loss and gradients
        self.q_optim.zero_grad()
        td_error = self.__calc_td_error(obs, act, rew, obs_next, done)
        loss_weights = self.buf.get_weights(self.device)
        loss_q = ((td_error * loss_weights)**2).mean()
        loss_q.backward()

        # Clip gradients (if neccesary) and update parameters
        if self.clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.q_net_mod.parameters(), 
                                           self.max_grad_norm)
        self.q_optim.step()
        self.buf.update_priorities(td_error.detach().cpu().numpy())

        # Update target Q network values if neccessary
        if update_target_network == True:
            self.q_net_mod.update_target()

        # Log epoch statistics
        self.writer.add_scalar('Loss/LossQ', loss_q.item(), epoch+1)
    
    def train_mod(self, epochs=100):
        # Initialize beta rate if buffer is a PRB
        if isinstance(self.buf, PrioritizedReplayBuffer):
            self.buf.beta_rate = (1.0 - self.buf.beta)/(epochs-1) \
                if self.buf.beta_rate is None else self.buf.beta_rate
        
        # Initialize scheduler
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        q_scheduler = LinearLR(self.q_optim, start_factor=1.0, end_factor=end_factor, 
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
                act, q_val = self.q_net_mod.step(torch.as_tensor(obs, dtype=torch.float32, 
                                                                 device=self.device))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                               q_val[env_id], terminated[env_id])
                        q_vals.append(q_val[env_id])
                obs = obs_next
                autoreset = np.logical_or(terminated, truncated)
                
                if (self.buf.get_buffer_size() >= self.learning_starts) \
                    and ((step % self.train_freq) == 0):
                    global_step = (step + epoch * self.steps_per_epoch) * self.env.num_envs
                    update_target_network = (global_step % self.target_network_update_freq) == 0
                    self.__update_params(epoch, update_target_network)

            # Evaluate deterministic policy (skip return normalization wrapper)
            env = self.env.env if isinstance(self.env, NormalizeReward) else self.env 
            ep_len, ep_ret = np.zeros(env.num_envs), np.zeros(env.num_envs)
            ep_lens, ep_rets = [], []
            obs, _ = env.reset()
            while len(ep_lens) < self.num_test_episodes*env.num_envs:
                act = self.q_net_mod.act(torch.as_tensor(obs, dtype=torch.float32, 
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

            if (epoch % self.save_freq) == 0:
                torch.save(self.q_net_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.q_net_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))

            q_scheduler.step()
            self.q_net_mod.update_eps_exp()
            self.buf.update_beta()
            
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
            self.writer.add_scalar('Time', time.time()-start_time, epoch+1)
            self.writer.flush()
            q_vals = []
        
        # Save final model
        torch.save(self.q_net_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')

    # Model generic arguments
    parser.add_argument('--dueling', type=bool, default=False)
    parser.add_argument('--double_q', type=bool, default=False)
    parser.add_argument('--prioritized_replay', type=bool, default=False)
    parser.add_argument('--eps_init', type=float, default=1.0)
    parser.add_argument('--eps_final', type=float, default=0.05)
    parser.add_argument('--eps_decay_rate', type=float, default=0.2)

    # MLP model arguments
    parser.add_argument('--hid', nargs='+', type=int, default=[64, 64])
    
    # CNN model arguments
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', nargs='+', type=int, default=[32, 64, 64])
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[8, 4, 3])
    parser.add_argument('--strides', nargs='+', type=int, default=[4, 2, 1])
    parser.add_argument('--features_out', nargs='+', type=int, default=[512])
    
    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--target_network_update_freq', type=int, default=500)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=bool, default=True)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Remove existing logs if run already exists
    clear_logs(log_dir)

    # Determine type of policy and setup its arguments and environment
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if args.policy == 'mlp':
        q_net_kwargs = dict(hidden_sizes=args.hid,
                            hidden_acts=torch.nn.ReLU)
        env_fn = [lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                    render_mode=render_mode)] * args.cpu
        wrappers_kwargs = dict()
    elif args.policy == 'cnn':
        q_net_kwargs = dict(in_channels=args.in_channels, 
                            out_channels=args.out_channels,
                            kernel_sizes=args.kernel_sizes, 
                            strides=args.strides, 
                            features_out=args.features_out)
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
    trainer = DQNTrainer(args.policy, env_fn, wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                         model_path=args.model_path, dueling=args.dueling, double_q=args.double_q, 
                         q_net_kwargs=q_net_kwargs, seed=args.seed, prioritized_replay=args.prioritized_replay, 
                         eps_init=args.eps_init, eps_final=args.eps_final, eps_decay_rate=args.eps_decay_rate, 
                         buf_size=args.buf_size, steps_per_epoch=args.steps, batch_size=args.batch_size,
                         learning_starts=args.learning_starts, train_freq=args.train_freq, 
                         target_network_update_freq=args.target_network_update_freq, 
                         num_test_episodes=args.num_test_episodes, gamma=args.gamma, lr=args.lr,
                         lr_f=args.lr_f, max_grad_norm=args.max_grad_norm, clip_grad=args.clip_grad, 
                         log_dir=log_dir, save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
    
    trainer.train_mod(args.epochs)