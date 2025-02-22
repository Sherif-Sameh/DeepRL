import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from core.dqn.models.mlp import MLPDQN, MLPDuelingDQN, MLPDDQN, MLPDuelingDDQN
from core.dqn.models.cnn import CNNDQN, CNNDuelingDQN, CNNDDQN, CNNDuelingDDQN
from core.rl_utils import SkipAndScaleObservation, NormalizeRewardManual, \
                        NormalizeObservationManual, NormalizeObservationFrozen
from core.rl_utils import save_env, run_env, update_mean_var_env
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
        self.done = np.zeros((env.num_envs, self.env_buf_size), dtype=np.bool)

        # Initialize array for storing the episode number of each experience tuple collected
        self.ep_nums = np.zeros((env.num_envs, self.env_buf_size), dtype=np.int64)
        self.ep_num_ctrs = np.zeros(env.num_envs, dtype=np.int64)

    def update_buffer(self, env_id, obs, act, rew, done):
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
        self.rew[env_id, self.ctr[env_id]] = rew
        self.done[env_id, self.ctr[env_id]] = done
        self.ep_nums[env_id, self.ctr[env_id]] = self.ep_num_ctrs[env_id]

        # Update buffer counter and reset if neccessary
        self.ctr[env_id] += 1
        if self.ctr[env_id] == self.env_buf_size:
            self.ctr[env_id] = 0
            self.buf_full[env_id] = True
            self.ep_num_ctrs[env_id] += 1

    def get_batch(self):
        num_envs = self.obs.shape[0]
        
        # Generate random indices for environments and experiences
        env_indices = np.random.randint(0, num_envs, size=self.batch_size)
        size = np.where(self.buf_full == True, self.env_buf_size, self.ctr)
        exp_indices = np.random.choice(np.min(size)-1, self.batch_size, replace=False)
        
        # Sample these random experiences from the replay buffer
        obs = self.obs[env_indices, exp_indices]
        act = self.act[env_indices, exp_indices]
        rew = self.rew[env_indices, exp_indices]
        obs_next = self.obs[env_indices, exp_indices+1]
        done = self.done[env_indices, exp_indices]
        mask = self.ep_nums[env_indices, exp_indices] == self.ep_nums[env_indices, exp_indices+1]

        # Return randomly selected experience tuples
        return obs, act, rew, obs_next, done, mask
    
    def get_buffer_size(self):
        return np.mean(np.where(self.buf_full, self.env_buf_size, self.ctr))
    
    def terminate_ep(self, env_id):
        self.ep_num_ctrs[env_id] += 1

    def get_weights(self, device):
        return torch.ones(self.batch_size, dtype=torch.float32, device=device)
    
    def update_beta(self):
        pass
    
    def update_priorities(self, td_errors):
        pass

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

    def update_buffer(self, env_id, obs, act, rew, done):
        # Normal replay buffer updates
        self.obs[env_id, self.ctr[env_id]] = obs
        self.act[env_id, self.ctr[env_id]] = act
        self.rew[env_id, self.ctr[env_id]] = rew
        self.done[env_id, self.ctr[env_id]] = done
        self.ep_nums[env_id, self.ctr[env_id]] = self.ep_num_ctrs[env_id]
        
        # Set priorities of new transitions to max priority
        self.pri[env_id, self.ctr[env_id]] = self.pri_max[env_id]

        # Update buffer counter and reset if neccessary
        self.ctr[env_id] += 1
        if self.ctr[env_id] == self.env_buf_size:
            self.ctr[env_id] = 0
            self.buf_full[env_id] = True
            self.ep_num_ctrs[env_id] += 1

    def get_batch(self):
        env_bs = self.batch_size // self.obs.shape[0]

        # Initialize empty batches for storing samples from the environments
        obs = np.zeros((self.batch_size,)+self.obs_shape, dtype=np.float32)
        act = np.zeros((self.batch_size,)+self.act_shape, dtype=np.float32)
        rew = np.zeros(self.batch_size, dtype=np.float32)
        obs_next = np.zeros((self.batch_size,)+self.obs_shape, dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.bool)
        mask = np.zeros(self.batch_size, dtype=np.bool)
        
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
            mask[env_slice] = (self.ep_nums[env_id, self.indices] == self.ep_nums[env_id, self.indices+1])

        # Return randomly selected experience tuples
        return obs, act, rew, obs_next, done, mask

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
    """ 
    Deep Q-learning Network (DQN) 
    
    :param policy: A string specifying the type of policy to be used ('mlp' or 'cnn').
    :param env_fn: A list of duplicated callable functions that are each used to initialize 
        an instance of the environment to use in training. The number of entries determines
        the number of parallel environment used in training.
    :param wrappers_kwargs: A dictionary of dictionaries where each key corresponds to the 
        class name of a wrapper applied to the environment. Each value corresponds to the 
        dictionary of key-value arguments passed to that wrapper upon initialization. This
        is required for saving environments and reloading them for testing. 
    :param use_gpu: Boolean flag that if set we'll attempt to use the Nvidia GPU for model
        evaluation and training if available. Otherwise the CPU is used by default.
    :param model_path: Absolute path to an existing AC model to load initial parameters from.
    :param dueling: Boolean flag that specifies if a dueling DQN architecture should be used.
    :param double_q: Boolean flag that specifies if double DQN should be used. 
    :param q_net_kwargs: A dictionary of key-value arguments to pass to the DQN's class 
        constructor. All arguments other than a reference to the env and Epsilon-Greedy 
        parameters are passed in this way. 
    :param seed: Seed given to RNGs. Set for everything (NumPy, PyTorch and CUDA if needed)
    :param prioritized_replay: Boolean flat that specifies if Prioritized Experience Replay (PER)
        should be used.
    :param prioritized_replay_alpha: Alpha parameter used by PER for calculating probabilites 
        for each sample based on its priority. Higher values prioritize large TD errors more.
    :param prioritized_replay_beta0: Initial value of PER's beta parameter. Used to compensate
        for bias introduced by prioritized sampling. Always annealed to 1.0 (unbiased).  
    :param prioritized_replay_beta_rate: Rate for annealing PER's beta parameter. Automatically
        calculated if left to 0, such that beta reaches 1.0 by the final epoch.
    :param prioritized_replay_eps: Small epsilon value used for priority calculated to ensure 
        that all samples have a non-zero probability of being sampled. 
    :param eps_init: Initial value for epsilon parameter used for the Epsilon-Greedy policy.
    :param eps_final: Final value for epsilon parameter used for the Epsilon-Greedy policy.
    :param eps_decay_epochs: The number of epochs over which the epsilon paramter used for 
        Epsilon-Greedy will be annealed from its initial to its final value. If not set, then 
        20% of all training epochs is used by default. Epsilon is always decayed exponentially.
    :param buf_size: Total size of the experience replay buffer in terms of the number of
        experience tuples stored. 
    :param steps_per_epoch: The number of steps executed in the environment per rollout before
        an offline policy evaluation step and peformance logging take place. Used per environment 
        when running multiple environments in parallel.
    :param batch_size: Batch size used for sampling experiences from the experience replay buffer. 
        Used per environment, so if batch_size=100 and there are 4 parallel environments, 400
        experience tuples are sampled from the buffer for each update.
    :param learning_starts: Number of steps to take in the environment to populate the replay 
        buffer before parameter updates can start taking place. Also defined per environment
        when running multiple environments in parallel.
    :param train_freq: Number of steps to take in the environment before a parameter update step.
        Also defined per environment when running multiple environments in parallel.
    :param num_updates: Number of sequential parameter updates per update step.
    :param target_network_update_freq: Number of steps to take in the environment before the 
        parameters of the main network and copied into the target network. Also defined per 
        environment when running multiple environments in parallel.
    :param num_test_episodes: Number of episodes to run the greedy policy for during offline 
        policy evaluation at the end of an epoch. Also defined per environment when running 
        multiple environments in parallel.
    :param gamma: The discount factor used for future rewards. 
    :param lr: Initial learning rate for ADAM optimizer.
    :param lr_f: Final learning rate for LR scheduling, using a linear schedule, if provided.
    :param norm_rew: Boolean flag that determines whether to apply return normalization or not.
    :param norm_obs: Boolean flag that determines whether to apply observation normalization or not.
    :param max_grad_norm: Upper limit used for limiting the model's combined parameters' 
        gradient norm. 
    :param clip_grad: Boolean flag that determines whether to apply gradient norm clipping or not.
    :param log_dir: Absolute path to the directory to use for storing training logs, models and 
        environements. Created if it does not already exist. Note that previous logs are deleted
        if an existing log directory is used. 
    :param save_freq: Number of epochs after which the current AC model is saved, overriding previous
        existing models. 
    :param checkpoint_freq: Number of epochs after which the current AC model is saved as an independent
        checkpoint model that will not be overriden in the future. 
    """
    def __init__(self, policy, env_fn, wrappers_kwargs=dict(), use_gpu=False, model_path='', 
                 dueling=False, double_q=False, q_net_kwargs=dict(), seed=0, 
                 prioritized_replay=False, prioritized_replay_alpha=0.6, 
                 prioritized_replay_beta0=0.4, prioritized_replay_beta_rate=0.0, 
                 prioritized_replay_eps=1e-6, eps_init=1.0, eps_final=0.05, 
                 eps_decay_epochs=None, buf_size=1000000, steps_per_epoch=1000, 
                 batch_size=100, learning_starts=1000, train_freq=4, num_updates=1, 
                 target_network_update_freq=500, num_test_episodes=10, gamma=0.99, 
                 lr=5e-4, lr_f=None, norm_rew=False, norm_obs=False, max_grad_norm=0.5, 
                 clip_grad=False, log_dir=None, save_freq=10, checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.num_updates = num_updates 
        self.target_network_update_freq = target_network_update_freq
        self.num_test_episodes = num_test_episodes
        self.epochs = None
        self.gamma = gamma
        self.lr = lr
        self.lr_f = lr_f
        self.norm_rew = norm_rew
        self.norm_obs = norm_obs
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad
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
        if self.norm_obs:
            self.env = NormalizeObservationManual(self.env)
            env_fn_save = lambda render_mode=None: NormalizeObservationFrozen(env_fn[0](render_mode=render_mode))
            wrappers_kwargs['NormalizeObservationFrozen'] = {}
        else:
            env_fn_save = env_fn[0]
        try:
            save_env(env_fn_save, wrappers_kwargs, self.writer.get_logdir(), render_mode='human')
            save_env(env_fn_save, wrappers_kwargs, self.writer.get_logdir(), render_mode='rgb_array')
        except Exception as e:
            print(f'Could not save environment: {e} \n\n')
        
        # Initialize Q-network
        if len(model_path) > 0:
            self.q_net_mod = torch.load(model_path, weights_only=False)
            self.q_net_mod = self.q_net_mod.to(torch.device('cpu'))
        else:
            eps_decay_rate = -np.log(eps_final/eps_init) / eps_decay_epochs \
                if eps_decay_epochs is not None else None
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
        
    def _calc_td_error(self, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor, 
                        obs_next: torch.Tensor, done: torch.Tensor, mask: torch.Tensor):
        # Get current and target Q vals and mask target Q vals where done is True
        q_vals = self.q_net_mod.forward_grad(obs, act)
        q_vals_target = self.q_net_mod.forward_target(obs_next)
        q_vals_target[done] = 0.0

        # Calculate TD target and error
        td_target = rew + self.gamma * q_vals_target
        td_error = td_target - q_vals

        return td_error[mask]
    
    def _update_params(self, epoch):
        # Get mini-batch and pre-process it
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype, device=self.device)
        obs, act, rew, obs_next, done, mask = self.buf.get_batch()
        if self.norm_obs:
            obs, obs_next = self._normalize_obs(obs), self._normalize_obs(obs_next)
        if self.norm_rew:
            rew = self._normalize_rew(rew)
        obs, act, rew, obs_next, done, mask = to_tensor(obs, torch.float32), to_tensor(act, torch.float32), \
                                            to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                                            to_tensor(done, torch.bool), to_tensor(mask, torch.bool)

        # Calculate Q-function loss and gradients
        self.q_optim.zero_grad()
        td_error = self._calc_td_error(obs, act, rew, obs_next, done, mask)
        loss_weights = self.buf.get_weights(self.device)[mask]
        loss_q = ((td_error * loss_weights)**2).mean()
        loss_q.backward()

        # Clip gradients (if neccesary) and update parameters
        if self.clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.q_net_mod.parameters(), 
                                           self.max_grad_norm)
        self.q_optim.step()
        self.buf.update_priorities(td_error.detach().cpu().numpy())

        # Log epoch statistics
        self.writer.add_scalar('Loss/LossQ', loss_q.item(), epoch+1)

    def _normalize_obs(self, obs):
        env = self.env
        while not hasattr(env, 'normalize_observations'):
            env = env.env
        
        return env.normalize_observations(obs)
    
    def _normalize_rew(self, rew):
        return self.env.normalize_rewards(rew)
    
    def _proc_env_rollout(self, env_id, ep_len):
        self.buf.terminate_ep(env_id)
    
    def _train(self, epoch):
        for _ in range(self.num_updates):
            self._update_params(epoch)

    def _eval(self):
        # Evaluate deterministic policy 
        ep_len, ep_ret = np.zeros(self.env.num_envs), np.zeros(self.env.num_envs)
        self.ep_len_list, self.ep_ret_list = [], []
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
        obs, _ = self.env.reset()
        while len(self.ep_len_list) < self.num_test_episodes*self.env.num_envs:
            if self.norm_obs: obs = self._normalize_obs(obs)
            act = self.q_net_mod.act(to_tensor(obs))
            obs, rew, terminated, truncated, _ = self.env.step(act)
            ep_len, ep_ret = ep_len + 1, ep_ret + rew
            done = np.logical_or(terminated, truncated)
            for env_id in range(self.env.num_envs):
                if done[env_id]:
                    self.ep_len_list.append(ep_len[env_id])
                    self.ep_ret_list.append(ep_ret[env_id])
                    ep_len[env_id], ep_ret[env_id] = 0, 0
    
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
        self.writer.add_scalar('Epsilon', self.q_net_mod.eps, epoch+1)

    def _save_model(self, epoch):
        if ((epoch + 1) % self.save_freq) == 0:
            torch.save(self.q_net_mod, os.path.join(self.save_dir, 'model.pt'))
            if self.norm_obs:
                update_mean_var_env(self.env, self.writer.get_logdir(), render_mode='human')
        if ((epoch + 1) % self.checkpoint_freq) == 0:
            torch.save(self.q_net_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))

    def _end_training(self):
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        if self.norm_obs:
            update_mean_var_env(self.env, self.writer.get_logdir(), render_mode='human')
            update_mean_var_env(self.env, self.writer.get_logdir(), render_mode='rgb_array')
        self.writer.close()
        self.env.close()

    def learn(self, epochs=100, ep_init=10):
        # Initialize epsilon decay rate if not initialized
        if self.q_net_mod.eps_decay_rate is None:
            eps_decay_epochs = 0.2 * epochs
            self.q_net_mod.eps_decay_rate = -np.log(self.q_net_mod.eps_min/self.q_net_mod.eps) / eps_decay_epochs

        # Initialize beta rate if buffer is a PRB
        if isinstance(self.buf, PrioritizedReplayBuffer):
            self.buf.beta_rate = (1.0 - self.buf.beta)/(epochs-1) \
                if self.buf.beta_rate is None else self.buf.beta_rate
        
        # Initialize scheduler
        self.epochs = epochs
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        q_scheduler = LinearLR(self.q_optim, start_factor=1.0, end_factor=end_factor, 
                               total_iters=epochs)
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Normalize returns for more stable training
        if self.norm_rew == True:
            self.env = NormalizeRewardManual(self.env, gamma=self.gamma)
        self.env.reset(seed=self.seed)
        run_env(self.env, num_episodes=ep_init)

        # Initialize environment variables
        obs, _ = self.env.reset()
        ep_len = np.zeros(self.env.num_envs, dtype=np.int64)
        autoreset = np.zeros(self.env.num_envs)
        q_val_list = []

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                act, q_val = self.q_net_mod.step(to_tensor(self._normalize_obs(obs) if self.norm_obs else obs))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)

                for env_id in range(self.env.num_envs):
                    self.buf.update_buffer(env_id, obs[env_id], act[env_id], 
                                            rew[env_id], terminated[env_id])
                    if not autoreset[env_id]:
                        q_val_list.append(q_val[env_id])
                        ep_len[env_id] += 1
                    else:
                        self._proc_env_rollout(env_id, ep_len)
                        ep_len[env_id] = 0 
                obs = obs_next
                autoreset = np.logical_or(terminated, truncated)
                
                if (self.buf.get_buffer_size() >= self.learning_starts) \
                    and ((step % self.train_freq) == 0):
                    self._train(epoch)

                if (step % self.target_network_update_freq) == 0:
                    self.q_net_mod.update_target()

            self._eval()
            obs, _ = self.env.reset()
            q_scheduler.step()
            self.q_net_mod.update_eps_exp()
            self.buf.update_beta()
            self._save_model()

            # Log info about epoch
            self._log_ep_stats(epoch, q_val_list)
            q_val_list = []
            self.writer.flush()
        
        # Save final model and finalize training
        self._end_training()
        print(f'Model {epochs} (final) saved successfully')

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument('--model_path', type=str, default='')

    # Model generic arguments
    parser.add_argument('--dueling', action="store_true", default=False)
    parser.add_argument('--double_q', action="store_true", default=False)
    parser.add_argument('--prioritized_replay', action="store_true", default=False)
    parser.add_argument('--eps_init', type=float, default=1.0)
    parser.add_argument('--eps_final', type=float, default=0.05)
    parser.add_argument('--eps_decay_epochs', type=int, default=None)

    # MLP model arguments
    parser.add_argument('--hid', nargs='+', type=int, default=[64, 64])
    
    # CNN model arguments
    parser.add_argument('--action_rep', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', nargs='+', type=int, default=[32, 64, 64])
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[8, 4, 3])
    parser.add_argument('--strides', nargs='+', type=int, default=[4, 2, 1])
    parser.add_argument('--padding', nargs='+', type=int, default=[0, 0, 0])
    parser.add_argument('--features_out', nargs='+', type=int, default=[512])
    
    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--ep_init', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--num_updates', type=int, default=1)
    parser.add_argument('--target_network_update_freq', type=int, default=500)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--norm_rew', action="store_true", default=False)
    parser.add_argument('--norm_obs', action="store_true", default=False)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', action="store_true", default=False)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='dqn')
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
                            padding=args.padding,
                            features_out=args.features_out)
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
    trainer = DQNTrainer(args.policy, env_fn, wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                         model_path=args.model_path, dueling=args.dueling, double_q=args.double_q, 
                         q_net_kwargs=q_net_kwargs, seed=args.seed, prioritized_replay=args.prioritized_replay, 
                         eps_init=args.eps_init, eps_final=args.eps_final, eps_decay_epochs=args.eps_decay_epochs, 
                         buf_size=args.buf_size, steps_per_epoch=args.steps, batch_size=args.batch_size,
                         learning_starts=args.learning_starts, train_freq=args.train_freq, 
                         num_updates=args.num_updates, target_network_update_freq=args.target_network_update_freq, 
                         num_test_episodes=args.num_test_episodes, gamma=args.gamma, lr=args.lr,
                         lr_f=args.lr_f, norm_rew=args.norm_rew, norm_obs=args.norm_obs, 
                         max_grad_norm=args.max_grad_norm, clip_grad=args.clip_grad, log_dir=log_dir, 
                         save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
    
    trainer.learn(epochs=args.epochs, ep_init=args.ep_init)