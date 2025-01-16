import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers.vector import RescaleAction, ClipAction, NormalizeReward
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.ppo.models.mlp import MLPActorCritic
from core.ppo.models.cnn import CNNActorCritic
from core.ppo.models.cnn_lstm import CNNLSTMActorCritic
from core.rl_utils import SkipAndScaleObservation, save_env, run_env
from core.utils import serialize_locals, clear_logs

class PPOBuffer:
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size, gamma, lam):
        # Check the type of the action space
        if not (isinstance(env.single_action_space, Discrete) or 
                isinstance(env.single_action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.gamma, self.lam = gamma, lam
        self.buf_size, self.batch_size = buf_size, batch_size
        self.ep_start = np.zeros(env.num_envs, dtype=np.int64)
        self.obs_shape = env.single_observation_space.shape
        self.act_shape = env.single_action_space.shape
        env_buf_size = buf_size // env.num_envs

        # Initialize all buffers for storing data during an epoch and training
        self.obs = np.zeros((env.num_envs, env_buf_size) + self.obs_shape, dtype=np.float32)
        self.act = np.zeros((env.num_envs, env_buf_size) + self.act_shape, dtype=np.float32)
        self.rew = np.zeros((env.num_envs, env_buf_size+1), dtype=np.float32)
        self.rtg = np.zeros((env.num_envs, env_buf_size), dtype=np.float32)
        self.adv = np.zeros((env.num_envs, env_buf_size), dtype=np.float32)
        self.val = np.zeros((env.num_envs, env_buf_size), dtype=np.float32)
        self.logp = np.zeros((env.num_envs, env_buf_size), dtype=np.float32)

    def update_buffer(self, env_id, obs, act, rew, val, logp, step):
        self.obs[env_id, step] = obs
        self.act[env_id, step] = act
        self.rew[env_id, step] = rew
        self.val[env_id, step] = val
        self.logp[env_id, step] = logp

    def terminate_ep(self, env_id, ep_len, val_terminal):
        ep_start, ep_end = self.ep_start[env_id], self.ep_start[env_id]+ep_len
        ep_slice = slice(ep_start, ep_end)

        # Calculate per episode statistics - Return to Go 
        self.rtg[env_id, ep_end-1] = self.rew[env_id, ep_end-1] + self.gamma*val_terminal
        for i in range(ep_len-2, -1, -1):
            self.rtg[env_id, ep_start+i] = self.rew[env_id, ep_start+i] + self.gamma*self.rtg[env_id, ep_start+i+1]
                                               
        # Calculate per episode statistics - Advantage function (GAE)
        rews = np.append(self.rew[env_id, ep_slice], val_terminal)
        vals = np.append(self.val[env_id, ep_slice], val_terminal)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[env_id, ep_end-1] = deltas[-1]
        for i in range(ep_len-2, -1, -1):
            self.adv[env_id, ep_start+i] = deltas[i] + self.gamma * self.lam * self.adv[env_id, ep_start+i+1]
        
        ep_ret = np.sum(self.rew[env_id, ep_slice])
        self.ep_start[env_id] += ep_len
    
        return ep_ret
    
    def terminate_epoch(self):
        self.ep_start = np.zeros_like(self.ep_start)
        
    def get_dataloader(self, device):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32, device=device)
        dataset = TensorDataset(to_tensor(self.obs.reshape((-1,)+self.obs_shape)), 
                                to_tensor(self.act.reshape((-1,)+self.act_shape)), 
                                to_tensor(self.adv.reshape(-1)), 
                                to_tensor(self.logp.reshape(-1)),
                                to_tensor(self.rtg.reshape(-1)))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_all_obs(self, device):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32, device=device)

        return to_tensor(self.obs.reshape((-1,)+self.obs_shape))

class PPOSequenceBuffer(PPOBuffer, Dataset):
    def __init__(self, env: AsyncVectorEnv, buf_size, batch_size, 
                 gamma, lam, seq_len, seq_prefix, stride):
        super().__init__(env, buf_size, batch_size, gamma, lam)
        self.device = None
        self.seq_len = seq_len
        self.seq_prefix = seq_prefix
        self.stride = stride
        self.num_sequences_env = ((buf_size//env.num_envs) - seq_len) // stride + 1
        self.num_sequences = self.num_sequences_env * env.num_envs

        # Initialize array for storing the episode number of each experience tuple collected
        self.ep_nums = np.zeros((env.num_envs, buf_size//env.num_envs), dtype=np.int64)
        self.ep_num_ctrs = np.zeros(env.num_envs, dtype=np.int64)

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype, device=self.device)

        # Calculate start and environment indices from sequence index
        env_id = idx // self.num_sequences_env
        start_idx = (idx - env_id*self.num_sequences_env) * self.stride

        # Extract experience sequences
        obs_seq = to_tensor(self.obs[env_id, start_idx:start_idx+self.seq_len], torch.float32)
        act_seq = to_tensor(self.act[env_id, start_idx:start_idx+self.seq_len], torch.float32)
        adv_seq = to_tensor(self.adv[env_id, start_idx:start_idx+self.seq_len], torch.float32)
        logp_seq = to_tensor(self.logp[env_id, start_idx:start_idx+self.seq_len], torch.float32)
        rtg_seq = to_tensor(self.rtg[env_id, start_idx:start_idx+self.seq_len], torch.float32)
        seq_mask = to_tensor(self.ep_nums[env_id, start_idx:start_idx+self.seq_len]\
                             ==self.ep_nums[env_id, start_idx], torch.bool)
        seq_mask[:self.seq_prefix] = False
        
        return obs_seq, act_seq, adv_seq, logp_seq, rtg_seq, seq_mask

    def update_buffer(self, env_id, obs, act, rew, val, logp, step):
        super().update_buffer(env_id, obs, act, rew, val, logp, step)
        self.ep_nums[env_id, step] = self.ep_num_ctrs[env_id]
    
    def terminate_ep(self, env_id, ep_len, val_terminal):
        ep_ret = super().terminate_ep(env_id, ep_len, val_terminal)
        self.ep_num_ctrs[env_id] += 1

        return ep_ret
    
    def terminate_epoch(self):
        super().terminate_epoch()
        self.ep_num_ctrs = np.zeros_like(self.ep_num_ctrs)
    
    def get_dataloader(self, device):
        self.device = device

        return DataLoader(self, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def get_all_obs(self, device):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype, device=device)

        # Create mask to mask out values at the beginning of a new episode
        mask = np.full(self.obs.shape[:2], fill_value=True)
        for env_id in range(self.obs.shape[0]):
            ep_resets = np.nonzero(np.diff(self.ep_nums[env_id]) != 0)[0] + 1
            ep_resets = np.append(ep_resets, 0)
            mask_indices = np.array([np.arange(idx, idx+self.seq_prefix) for idx in ep_resets])
            mask_indices = np.minimum(mask_indices, mask.shape[1] - 1)
            mask[env_id, mask_indices] = False
        
        return to_tensor(self.obs, torch.float32), to_tensor(mask, torch.bool)
                

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO)

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
    :param ac: Class type that defines the type of policy to be used (MLP, CNN, etc)
    :param ac_kwargs: A dictionary of key-value arguments to pass to the actor-critic's class  
        contructor. All arguments other than a reference to the env are passed in this way.
    :param seed: Seed given to RNGs. Set for everything (NumPy, PyTorch and CUDA if needed)
    :param steps_per_epoch: The number of steps executed in the environment per rollout before
        a parameter update step takes place. Used per environment when running multiple
        environments in parallel.
    :param eval_every: The number of epochs to pass between logging the agent's current 
        performance. Note that for on-policy algorithms, evaluation is online based on
        the same rollouts used for training. 
    :param batch_size: Batch size used for sampling experiences from the experience buffer. 
        Used per environment, so if batch_size=100 and there are 4 parallel environments, 400
        experience tuples are sampled from the buffer for each update.
    :param gamma: The discount factor used for future rewards. 
    :param clip_ratio: Clip ratio used PPO's clipping mechanism of its surrogate objective 
        function for the policy's loss. 
    :param lr: Initial learning rate for ADAM optimizer.
    :param lr_f: Final learning rate for LR scheduling, using a linear schedule, if provided.
    :param ent_coeff: Entropy coefficient for the combined AC loss function combining the 
        policy, value and entropy losses. 
    :param vf_coeff: Value function loss coefficient for the combined AC loss function combining 
        the policy, value and entropy losses.
    :param max_grad_norm: Upper limit used for limiting the model's combined parameters' gradient 
        norm. Applied to both actor and critic's parameters together. 
    :param clip_grad: Boolean flag that determines whether to apply gradient norm clipping or not.
    :param train_iters: Number of loops over the whole experience buffer per a single parameter update.
    :param lam: The constant coefficient lambda used in GAE.
    :param target_kl: Soft-upper limit on allowable KL-Divergence between new and old policies.
        Early stopping of parameter updates is triggered if KL exceeds 1.5 * target_kl. 
    :param seq_len: The length of sequences used for training recurrent policies. 
    :param seq_prefix: The length of the 'burn-in' period used for stabilizing a recurrent policy's
        hidden states before feeding in the real sequence used for loss computation. Full sequence 
        length when sampling = seq_len + seq_prefix
    :param seq_stride: The stride (step size) between sequential sequences for recurrent policies. 
        Determines the amout of overlap between sequential sequences.
    :param log_dir: Absolute path to the directory to use for storing training logs, models and 
        environements. Created if it does not already exist. Note that previous logs are deleted
        if an existing log directory is used. 
    :param save_freq: Number of epochs after which the current AC model is saved, overriding previous
        existing models. 
    :param checkpoint_freq: Number of epochs after which the current AC model is saved as an independent
        checkpoint model that will not be overriden in the future. 
    """
    def __init__(self, env_fn, wrappers_kwargs=dict(), use_gpu=False, model_path='', 
                 ac=MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=1000,
                 eval_every=4, batch_size=100, gamma=0.99, clip_ratio=0.2, lr=3e-4, 
                 lr_f=None, ent_coeff=0.0, vf_coeff=0.5, max_grad_norm=0.5, 
                 clip_grad=False, train_iters=40, lam=0.95, target_kl=None, seq_len=80, 
                 seq_prefix=40, seq_stride=20, log_dir=None, save_freq=10, 
                 checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.eval_every = eval_every
        self.epochs = None
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.lr_f = lr_f
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.checkpoint_freq = checkpoint_freq
        self.ep_len_list, self.ep_ret_list = [], []

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
        self.env = ClipAction(RescaleAction(self.env, min_action=-1.0, max_action=1.0)) \
            if isinstance(self.env.single_action_space, Box) else self.env # Rescale cont. action spaces to [-1, 1]
        try:
            save_env(env_fn[0], wrappers_kwargs, self.writer.get_logdir(), render_mode='human')
            save_env(env_fn[0], wrappers_kwargs, self.writer.get_logdir(), render_mode='rgb_array')
        except Exception as e:
            print(f'Could not save environment: {e} \n\n')
        
        # Initialize actor-critic
        if len(model_path) > 0:
            self.ac_mod = torch.load(model_path, weights_only=False)
            self.ac_mod = self.ac_mod.to(torch.device('cpu'))
        else:
            self.ac_mod = ac(self.env, **ac_kwargs)
        self.ac_mod.layer_summary()
        self.writer.add_graph(self.ac_mod, torch.randn(size=self.env.observation_space.shape, dtype=torch.float32))

        # Setup random seed number for PyTorch and NumPy
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

        # GPU setup if necessary
        if use_gpu == True:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.manual_seed(seed=seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.ac_mod.to(self.device)

        # Initialize the experience buffer for training
        if ac != CNNLSTMActorCritic:
            self.buf = PPOBuffer(self.env, steps_per_epoch * self.env.num_envs, 
                                 batch_size * self.env.num_envs, gamma, lam)
        else:
            self.buf = PPOSequenceBuffer(self.env, steps_per_epoch * self.env.num_envs,
                                         batch_size * self.env.num_envs, gamma, lam,
                                         seq_len + seq_prefix, seq_prefix, seq_stride)

        # Initialize optimizer
        self.ac_optim = Adam(self.ac_mod.parameters(), lr=lr)

    def _calc_policy_loss(self, obs: torch.Tensor, act: torch.Tensor, adv: torch.Tensor, 
                           logp: torch.Tensor, loss_mask: torch.Tensor):
        log_prob = self.ac_mod.actor.log_prob_grad(obs, act)
        ratio = torch.exp(log_prob - logp)
        clipped_ratio = torch.clamp(ratio, 
                                    1 - self.clip_ratio,
                                    1 + self.clip_ratio)
        surrogate_obj = (torch.minimum(ratio * adv, other=(clipped_ratio * adv)))
        surrogate_obj = -(surrogate_obj[loss_mask]).mean()
    
        return surrogate_obj
    
    def _calc_entropy_loss(self, loss_mask: torch.Tensor):
        entropy = self.ac_mod.actor.entropy()
        entropy_loss = -(entropy[loss_mask].mean())
        
        return entropy_loss

    def _calc_val_loss(self, obs: torch.Tensor, rtg: torch.Tensor,
                        loss_mask: torch.Tensor):
        val = self.ac_mod.critic.forward(obs)
        val_loss = F.mse_loss(val[loss_mask], rtg[loss_mask])
        
        return val_loss 
    
    def _update_params(self, epoch):
        # Store old policy and all observations for KL early stopping
        obs_all = self.buf.get_all_obs(self.device) # Returns a tuple (obs, mask) for LSTM policies
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
        with torch.no_grad(): self.ac_mod.actor.forward(obs_all[0] if isinstance(obs_all, tuple) else obs_all)
        pi_curr = self.ac_mod.actor.copy_policy()

        # Loop train_iters times over the whole dataset (unless early stopping occurs)
        kls = []
        for _ in range(self.train_iters):
            # Get dataloader for performing mini-batch SGD updates
            dataloader = self.buf.get_dataloader(self.device)
            
            # Loop over dataset in mini-batches
            for batch in dataloader:
                # Clear gradients
                self.ac_optim.zero_grad()

                # Unpack experience tuple
                if len(batch) == 6:
                    obs, act, adv, logp, rtg, mask = batch
                else:
                    obs, act, adv, logp, rtg = batch
                    mask = torch.full((obs.shape[0],), fill_value=True)
                
                # Normalize advantages mini-batch wise
                adv_mean, adv_std = adv.mean(), adv.std()
                adv = (adv - adv_mean) / adv_std

                # Get policy, entropy and value losses
                self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
                loss_pi = self._calc_policy_loss(obs, act, adv, logp, loss_mask=mask)
                loss_ent = self._calc_entropy_loss(loss_mask=mask)
                self.ac_mod.reset_hidden_states(self.device, batch_size=self.buf.batch_size)
                loss_val = self._calc_val_loss(obs, rtg, loss_mask=mask)

                # Combine losses and compute gradients
                loss = loss_pi + self.ent_coeff * loss_ent + self.vf_coeff * loss_val
                loss.backward()

                # Clip gradients (if required) and update parameters
                if self.clip_grad == True:
                    torch.nn.utils.clip_grad_norm_(self.ac_mod.parameters(), self.max_grad_norm)
                self.ac_optim.step()

            # Check KL-Divergence constraint for triggering early stopping
            self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
            with torch.no_grad(): kl = self.ac_mod.actor.kl_divergence(obs_all, pi_curr)
            kls.append(kl.item())
            if (self.target_kl is not None) and (kl > 1.5 * self.target_kl):
                break

        # Log epoch statistics
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        self.writer.add_scalar('Loss/LossEnt', loss_ent.item(), epoch+1)
        self.writer.add_scalar('Loss/LossV', loss_val.item(), epoch+1)
        self.writer.add_scalar('Pi/KL', np.mean(kls), epoch+1)

    def _proc_env_rollout(self, env_id, val_terminal, rollout_len, ep_ret, ep_len):
        ep_ret[env_id] += self.buf.terminate_ep(env_id, rollout_len, val_terminal)
        ep_ret[env_id] *= np.sqrt(self.env.return_rms.var + self.env.epsilon)
        self.ep_len_list.append(ep_len[env_id])
        self.ep_ret_list.append(ep_ret[env_id])
        ep_len[env_id], ep_ret[env_id] = 0, 0
        self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)

    def _proc_epoch_rollout(self, obs_final, ep_ret, ep_len):
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        for env_id in range(self.env.num_envs): 
            if ep_len[env_id] > 0: 
                val_terminal = self.ac_mod.get_terminal_value(to_tensor(obs_final), env_id)
                ep_ret[env_id] += self.buf.terminate_ep(env_id, min(ep_len[env_id], self.steps_per_epoch), 
                                                        val_terminal)
        self.buf.terminate_epoch()

    def _train(self, epoch):
        self.ac_mod.reset_hidden_states(self.device, save=True)
        self._update_params(epoch)
        self.ac_mod.reset_hidden_states(self.device, restore=True)

    def _log_ep_stats(self, epoch):
        if len(self.ep_ret_list) > 0:
            total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
            ep_len_np, ep_ret_np = np.array(self.ep_len_list), np.array(self.ep_ret_list)
            self.writer.add_scalar('EpLen/mean', ep_len_np.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/mean', ep_ret_np.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/max', ep_ret_np.max(), total_steps_so_far)
            self.writer.add_scalar('EpRet/min', ep_ret_np.min(), total_steps_so_far)
        self.writer.add_scalar('VVals/mean', self.buf.val.mean(), epoch+1)
        self.writer.add_scalar('VVals/max', self.buf.val.max(), epoch+1)
        self.writer.add_scalar('VVals/min', self.buf.val.min(), epoch+1)
    
    def learn(self, epochs=100, ep_init=10):
        # Initialize scheduler
        self.epochs = epochs
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        ac_scheduler = LinearLR(self.ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
        # Normalize returns for more stable training
        if not isinstance(self.env, NormalizeReward):
            self.env = NormalizeReward(self.env, gamma=self.gamma)
        self.env.reset(seed=self.seed)
        run_env(self.env, num_episodes=ep_init)

        # Initialize environment variables
        obs, _ = self.env.reset()
        ep_len, ep_ret = np.zeros(self.env.num_envs, dtype=np.int64), np.zeros(self.env.num_envs)
        autoreset = np.zeros(self.env.num_envs)
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                act, val, logp = self.ac_mod.step(to_tensor(obs))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)
                autoreset_next = np.logical_or(terminated, truncated)
                ep_len += 1

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                               val[env_id], logp[env_id], step)
                    if autoreset_next[env_id]:
                        val_terminal = 0 if terminated[env_id] else self.ac_mod.get_terminal_value(to_tensor(obs_next), env_id)
                        self._proc_env_rollout(env_id, val_terminal, min(ep_len[env_id], step+1), ep_ret, ep_len)
                obs, autoreset = obs_next, autoreset_next

            self._proc_epoch_rollout(obs, ep_ret, ep_len)
            self._train(epoch)
            ac_scheduler.step()

            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))
                
            # Log info about epoch
            if ((epoch + 1) % self.eval_every) == 0:
                self._log_ep_stats(epoch)
                self.ep_len_list, self.ep_ret_list = [], []
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
    
    # MLP model arguments
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
    parser.add_argument('--log_std_init', nargs='+', type=float, default=[0]) # Used by all policies

    # CNN-LSTM specific model arguments
    parser.add_argument('--seq_len', type=int, default=80)
    parser.add_argument('--seq_prefix', type=int, default=40)
    parser.add_argument('--seq_stride', type=int, default=20)

    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ep_init', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--ent_coeff', type=float, default=0.0)
    parser.add_argument('--vf_coeff', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', action="store_true", default=False)
    parser.add_argument('--train_iters', type=int, default=10)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--target_kl', type=float, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='ppo')
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
                         hidden_acts_actor=torch.nn.Tanh, 
                         hidden_acts_critic=torch.nn.Tanh,
                         log_std_init=args.log_std_init)
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
                         log_std_init=args.log_std_init)
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
    trainer = PPOTrainer(env_fn, wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                         model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, seed=args.seed, 
                         steps_per_epoch=args.steps, eval_every=args.eval_every, 
                         batch_size=args.batch_size, gamma=args.gamma, clip_ratio=args.clip_ratio, 
                         lr=args.lr, lr_f=args.lr_f, ent_coeff=args.ent_coeff, 
                         vf_coeff=args.vf_coeff, max_grad_norm=args.max_grad_norm, 
                         clip_grad=args.clip_grad, train_iters=args.train_iters, 
                         lam=args.lam, target_kl=args.target_kl, seq_len=args.seq_len, 
                         seq_prefix=args.seq_prefix, seq_stride=args.seq_stride, log_dir=log_dir, 
                         save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
    
    trainer.learn(epochs=args.epochs, ep_init=args.ep_init)