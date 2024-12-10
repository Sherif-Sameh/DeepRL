import os
import inspect
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
import time
import numpy as np
import torch
import torch.distributions
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.mlp import MLPActorCritic
from models.cnn import CNNActorCritic

class SkipAndScaleObservation(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total_reward += rew
            if terminated or truncated:
                break

        return obs.astype(np.float32)/255.0, total_reward, terminated, truncated, info

def serialize_locals(locals_dict: dict):
    # Unpack dictionaries within locals_dict
    dict_keys = []
    for k in locals_dict:
        if isinstance(locals_dict[k], dict):
            dict_keys.append(k)
    for k in dict_keys:
        nested_dict = locals_dict.pop(k)
        for k_dict in nested_dict:
            locals_dict[k_dict] = nested_dict[k_dict]
    
    # Convert any value that is a class to its name and list to tensor
    for k in locals_dict:
        if inspect.isclass(locals_dict[k]):
            locals_dict[k] = locals_dict[k].__name__

        if isinstance(locals_dict[k], list):
            locals_dict[k] = torch.tensor(locals_dict[k])
    
    return locals_dict

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
        # Calculate per episode statistics - Return to Go 
        ep_start, ep_end = self.ep_start[env_id], self.ep_start[env_id]+ep_len
        self.rtg[env_id, ep_end-1] = self.rew[env_id, ep_end-1] + self.gamma*val_terminal
        for i in range(ep_len-2, -1, -1):
            self.rtg[env_id, ep_start+i] = self.rew[env_id, ep_start+i] + self.gamma*self.rtg[env_id, ep_start+i+1]
                                               
        # Calculate per episode statistics - Advantage function (GAE)
        ep_slice = slice(ep_start, ep_end)
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

    def get_pi_dataloader(self):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32)
        dataset = TensorDataset(to_tensor(self.obs.reshape((-1,)+self.obs_shape)), 
                                to_tensor(self.act.reshape((-1,)+self.act_shape)), 
                                to_tensor(self.adv.reshape(-1)), 
                                to_tensor(self.logp.reshape(-1)))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32)
        dataset = TensorDataset(to_tensor(self.obs.reshape((-1,)+self.obs_shape)), 
                                to_tensor(self.rtg.reshape(-1)))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            

class PPOTrainer:
    def __calc_policy_loss(self, actor, obs: torch.Tensor, act: torch.Tensor,
                           adv: torch.Tensor, logp: torch.Tensor, clip_ratio):
        log_prob = actor.log_prob_grad(obs, act)
        ratio = torch.exp(log_prob - logp)
        clipped_ratio = torch.clamp(ratio, 
                                    1 - clip_ratio,
                                    1 + clip_ratio)
        surrogate_obj = -(torch.min(ratio * adv, other=(clipped_ratio * adv))).mean()
        
        return surrogate_obj
    
    def __calc_val_loss(self, critic, obs: torch.Tensor, rtg: torch.Tensor):
        val = critic.forward_grad(obs)

        return ((val - rtg)**2).mean()
    
    def __update_params(self, device, ac_mod: MLPActorCritic, buf: PPOBuffer, ac_optim: Adam, 
                        writer: SummaryWriter, epoch, train_pi_iters, train_v_iters, 
                        clip_ratio, target_kl):
        # Store old policy for KL early stopping
        ac_mod.actor.update_policy(torch.as_tensor(buf.obs.reshape((-1,)+buf.obs_shape), 
                                                   dtype=torch.float32, device=device))
        if isinstance(ac_mod.actor.pi, torch.distributions.Categorical):
            pi_curr = torch.distributions.Categorical(logits=ac_mod.actor.pi.logits)
        elif isinstance(ac_mod.actor.pi, torch.distributions.Normal):
            pi_curr = torch.distributions.Normal(loc=ac_mod.actor.pi.mean, 
                                                 scale=ac_mod.actor.pi.stddev)
        
        # Loop train_pi_iters times over the whole dataset to update policy (unless early stopping occurs)
        for i in range(train_pi_iters):
            # Get dataloader for performing mini-batch SGD updates
            dataloader = buf.get_pi_dataloader()
            
            # Loop over dataset in mini-batches
            for obs, act, adv, logp in dataloader:
                obs, act, adv, logp = obs.to(device), act.to(device), adv.to(device), logp.to(device)

                # Normalize advantages mini-batch wise across all MPI processes
                adv_mean, adv_std = adv.mean(), adv.std()
                adv = (adv - adv_mean) / adv_std

                ac_optim.zero_grad()
                loss_pi = self.__calc_policy_loss(ac_mod.actor, obs, act,
                                                  adv, logp, clip_ratio)
                loss_pi.backward()
                ac_optim.step()
            
            # Check KL-Divergence constraint for triggering early stopping
            kl = ac_mod.actor.kl_divergence(torch.as_tensor(buf.obs.reshape((-1,)+buf.obs_shape), 
                                                            dtype=torch.float32, device=device), pi_curr)
            if kl > 1.5 * target_kl:
                print(f'Actor updates cut-off after {i+1} iterations by KL {kl}')
                break

        # Loop train_v_iters times over the whole dataset to update value function
        for i in range(train_v_iters):
            # Get dataloader for performing mini-batch SGD updates
            dataloader = buf.get_val_dataloader()
            
            # Loop over dataset in mini-batches
            for obs, rtg in dataloader:
                obs, rtg = obs.to(device), rtg.to(device)
                ac_optim.zero_grad()
                loss_val = self.__calc_val_loss(ac_mod.critic, obs, rtg)
                loss_val.backward()
                ac_optim.step()

        # Log epoch statistics
        writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        writer.add_scalar('Loss/LossV', loss_val.item(), epoch+1)
        writer.add_scalar('Pi/KL', kl.item(), epoch+1)

    
    def train_mod(self, env_fn, use_gpu=False, model_path='', ac=MLPActorCritic, 
                  ac_kwargs=dict(), seed=0, steps_per_epoch=4000, batch_size=400, 
                  epochs=50, gamma=0.99, clip_ratio=0.2, lr=3e-4, lr_f=None, 
                  train_pi_iters=80, train_v_iters=80, lam=0.97, target_kl=0.01, 
                  log_dir=None, save_freq=10, checkpoint_freq=25):
        # Serialize local hyperparameters
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn')
        locals_dict = serialize_locals(locals_dict)

        # Initialize logger and save hyperparameters
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(writer.get_logdir())}')
        save_dir = os.path.join(writer.get_logdir(), 'pyt_save')
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize environment and actor-critic
        env = AsyncVectorEnv(env_fn)
        if len(model_path) > 0:
            ac_mod = torch.load(model_path)
        else:
            ac_mod = ac(env, **ac_kwargs)
        ac_mod.layer_summary()
        writer.add_graph(ac_mod, torch.randn(size=env.observation_space.shape, dtype=torch.float32))

        local_steps_per_epoch = steps_per_epoch // env.num_envs

        # Setup random seed number for PyTorch and NumPy
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

        # GPU setup if necessary
        if use_gpu == True:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.manual_seed(seed=seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
        ac_mod.to(device)

        # Initialize the experience buffer for training
        buf_mod = PPOBuffer(env, steps_per_epoch, batch_size,
                            gamma, lam)

        # Initialize optimizer and scheduler
        ac_optim = Adam(ac_mod.parameters(), lr=lr)
        end_factor = lr_f/lr if lr_f is not None else 1.0
        ac_scheduler = LinearLR(ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)

        # Initialize environment variables
        obs, _ = env.reset(seed=seed)
        ep_len, ep_ret = np.zeros(env.num_envs, dtype=np.int64), 0
        ep_lens, ep_rets = [], []
        start_time = time.time()
        autoreset = np.zeros(env.num_envs)

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                act, val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32, device=device))
                obs_next, rew, terminated, truncated, _ = env.step(act)

                for env_id in range(env.num_envs):
                    if not autoreset[env_id]:
                        buf_mod.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                              val[env_id], logp[env_id], step)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_steps_per_epoch-1)
                autoreset = np.logical_or(terminated, truncated)

                if np.any(autoreset):
                    for env_id in range(env.num_envs):
                        if autoreset[env_id]:
                            val_terminal = 0 if terminated[env_id] else ac_mod.critic(
                                torch.as_tensor(obs[env_id], dtype=torch.float32, device=device)).cpu().numpy()
                            ep_ret = buf_mod.terminate_ep(env_id, ep_len[env_id], val_terminal)
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret)
                            ep_len[env_id] = 0
                
                if epoch_done:
                    obs, _ = env.reset()
                    buf_mod.terminate_epoch()
                    ep_len = np.zeros_like(ep_len)

            self.__update_params(device, ac_mod, buf_mod, ac_optim, writer, epoch, 
                                 train_pi_iters, train_v_iters, clip_ratio, target_kl)
            ac_scheduler.step()
            
            if (epoch % save_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
            if ((epoch + 1) % checkpoint_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, f'model{epoch+1}.pt'))
                
            
            # Log info about epoch
            if len(ep_rets) > 0:
                ep_lens, ep_rets = np.array(ep_lens), np.array(ep_rets)
                writer.add_scalar('EpLen/mean', ep_lens.mean(), (epoch+1)*steps_per_epoch)
                writer.add_scalar('EpRet/mean', ep_rets.mean(), (epoch+1)*steps_per_epoch)
                writer.add_scalar('EpRet/max', ep_rets.max(), (epoch+1)*steps_per_epoch)
                writer.add_scalar('EpRet/min', ep_rets.min(), (epoch+1)*steps_per_epoch)
                ep_lens, ep_rets = [], []
            writer.add_scalar('VVals/mean', buf_mod.val.mean(), epoch+1)
            writer.add_scalar('VVals/max', buf_mod.val.max(), epoch+1)
            writer.add_scalar('VVals/min', buf_mod.val.min(), epoch+1)
            writer.add_scalar('Time', time.time()-start_time, epoch+1)
            writer.flush()
        
        # Save final model
        torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
        writer.close()
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    
    # MLP model arguments
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
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_f', type=float, default=3e-4)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Set directory for logging
    log_dir = os.getcwd() + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Determine type of policy and setup its arguments and environment
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if args.policy == 'mlp':
        ac = MLPActorCritic
        ac_kwargs = dict(hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri,
                         hidden_acts_actor=torch.nn.Tanh, 
                         hidden_acts_critic=torch.nn.Tanh)
        env_fn = [lambda: gym.make(args.env, max_episode_steps=max_ep_len)] * args.cpu
    elif args.policy == 'cnn':
        ac = CNNActorCritic
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides, 
                         features_out=args.features_out)
        env_fn_def = lambda: gym.make(args.env, max_episode_steps=max_ep_len)
        env_fn = [lambda: FrameStackObservation(SkipAndScaleObservation(GrayscaleObservation(env_fn_def())), 
                                                stack_size=args.in_channels)] * args.cpu
    else:
        raise NotImplementedError
    
    # Begin training
    trainer = PPOTrainer()
    trainer.train_mod(env_fn, use_gpu=args.use_gpu, model_path=args.model_path, 
                      ac=ac, ac_kwargs=ac_kwargs, seed=args.seed, 
                      steps_per_epoch=args.steps, batch_size=args.batch_size, 
                      epochs=args.epochs, gamma=args.gamma, clip_ratio=args.clip_ratio, 
                      lr=args.lr, lr_f=args.lr_f,train_pi_iters=args.train_pi_iters, 
                      train_v_iters=args.train_v_iters, lam=args.lam, 
                      target_kl=args.target_kl, log_dir=log_dir, save_freq=args.save_freq, 
                      checkpoint_freq=args.checkpoint_freq)