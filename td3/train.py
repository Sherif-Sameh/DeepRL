import os
import inspect
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Box
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from models import MLPActorCritic, polyak_average

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

    def get_batch(self):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype)
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
        return np.sum(self.env_buf_size * self.buf_full + self.ctr * (1 - self.buf_full))
    
    
class TD3Trainer:
    def __calc_q_loss(self, ac: MLPActorCritic, gamma, target_noise, noise_clip, 
                      action_max, obs: torch.Tensor, act: torch.Tensor, 
                      rew: torch.Tensor, obs_next: torch.Tensor, done: torch.Tensor):
        # Get current Q values from both Q networks
        q_vals_1 = ac.critic_1.forward(obs, act)
        q_vals_2 = ac.critic_2.forward(obs, act)
        q_vals_1.squeeze_(dim=1)
        q_vals_2.squeeze_(dim=1)

        # Determine target Q values and mask values where done is True
        act_noise = torch.clamp(target_noise * torch.randn_like(act), 
                                min=-noise_clip, max=noise_clip)
        act_target = torch.max(torch.min(ac.actor.forward_target(obs_next) + act_noise, 
                                         action_max), -action_max)
        q_vals_target_1 = ac.critic_1.forward_target(obs_next, act_target)
        q_vals_target_2 = ac.critic_2.forward_target(obs_next, act_target)
        q_vals_target = torch.min(q_vals_target_1, q_vals_target_2)
        q_vals_target.squeeze_(dim=1)
        q_vals_target[done] = 0.0

        # Calculate TD target and errors
        td_target = rew + gamma * q_vals_target
        td_error_1 = td_target - q_vals_1
        td_error_2 = td_target - q_vals_2

        return (td_error_1**2).mean(), (td_error_2**2).mean()
    
    def __calc_pi_loss(self, ac: MLPActorCritic, obs: torch.Tensor):
        act = ac.actor.forward(obs)
        q_vals = ac.critic_1.forward(obs, act)
        q_vals.squeeze_(dim=1)

        return (-q_vals).mean()
    
    def __update_params(self, ac: MLPActorCritic, buf: ReplayBuffer, pi_optim: Adam, 
                        q1_optim: Adam, q2_optim: Adam, writer: SummaryWriter, epoch, 
                        gamma, polyak, target_noise, noise_clip, action_max, 
                        update_policy=False):
        # Get mini-batch from replay buffer
        obs, act, rew, obs_next, done = buf.get_batch()
        
        # Update critic networks
        q1_optim.zero_grad()
        q2_optim.zero_grad()
        loss_q1, loss_q2 = self.__calc_q_loss(ac, gamma, target_noise, noise_clip, 
                                              action_max, obs, act, rew, obs_next, 
                                              done)
        loss_q1.backward()
        loss_q2.backward()
        q1_optim.step()
        q2_optim.step()

        # Update target critic networks
        polyak_average(ac.critic_1.net.parameters(), ac.critic_1.net_target.parameters(), polyak)
        polyak_average(ac.critic_2.net.parameters(), ac.critic_2.net_target.parameters(), polyak)

        # Log training statistics
        writer.add_scalar('Loss/LossQ1', loss_q1.item(), epoch+1)
        writer.add_scalar('Loss/LossQ2', loss_q2.item(), epoch+1)

        if update_policy == True:
            # Update the actor network (critic's weights are frozen temporarily)
            ac.critic_1.set_grad_tracking(val=False)
            pi_optim.zero_grad()
            loss_pi = self.__calc_pi_loss(ac, obs)
            loss_pi.backward()
            pi_optim.step()
            ac.critic_1.set_grad_tracking(val=True)

            # Update target network
            polyak_average(ac.actor.net.parameters(), ac.actor.net_target.parameters(), polyak)

            # Log training statistics
            writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
    
    def train_mod(self, env_fn, model_path='', ac=MLPActorCritic, ac_kwargs=dict(), 
                  seed=0, steps_per_epoch=4000, epochs=100, buf_size=1000000, 
                  gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
                  batch_size=400, start_steps=10000, learning_starts=1000,
                  update_every=50, target_noise=0.2, noise_clip=0.5, 
                  policy_delay=2, num_test_episodes=10, log_dir=None, 
                  save_freq=10, checkpoint_freq=20):
        # Serialize local hyperparameters
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn')
        locals_dict = serialize_locals(locals_dict)

        # Initialize logger and save hyperparameters
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(writer.get_logdir())}')
        save_dir = os.path.join(writer.get_logdir(), 'pyt_save')
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize environment and Actor-Critic
        env = AsyncVectorEnv(env_fn)
        if len(model_path) > 0:
            ac_mod = torch.load(model_path)
        else:
            ac_mod = ac(env, **ac_kwargs)

        local_steps_per_epoch = steps_per_epoch // env.num_envs
        local_start_steps = start_steps // env.num_envs

        # Setup random seed number for PyTorch and NumPy
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # Initialize the experience replay buffer for training
        buf = ReplayBuffer(env, buf_size, batch_size)    

        # Store action limits for target action clipping
        action_max = torch.tensor(env.single_action_space.high) 

        # Initialize optimizers
        pi_optim = Adam(ac_mod.actor.net.parameters(), lr=pi_lr)
        q1_optim = Adam(ac_mod.critic_1.net.parameters(), lr=q_lr)
        q2_optim = Adam(ac_mod.critic_2.net.parameters(), lr=q_lr)

        # Initialize environment variables
        obs, _ = env.reset(seed=seed)
        start_time = time.time()
        autoreset = np.zeros(env.num_envs)
        q_vals = []

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                if (step + local_steps_per_epoch*epoch) > local_start_steps:
                    act, q_val = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    act = env.action_space.sample()
                    _, q_val = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                obs_next, rew, terminated, truncated, _ = env.step(act)

                for env_id in range(env.num_envs):
                    if not autoreset[env_id]:
                        buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                          q_val[env_id], terminated[env_id])
                        q_vals.append(q_val[env_id])
                obs = obs_next
                autoreset = np.logical_or(terminated, truncated)
                
                if (buf.get_buffer_size() >= learning_starts) \
                    and ((step % update_every) == 0):
                    for j in range(update_every):
                        update_policy = (j % policy_delay) == 0 
                        self.__update_params(ac_mod, buf, pi_optim, q1_optim, 
                                             q2_optim, writer, epoch, gamma, 
                                             polyak, target_noise, noise_clip, 
                                             action_max, update_policy=update_policy)
            
            # Evaluate deterministic policy
            ep_len, ep_ret = np.zeros(env.num_envs), np.zeros(env.num_envs)
            ep_lens, ep_rets = [], []
            obs, _ = env.reset()
            while len(ep_lens) < num_test_episodes*env.num_envs:
                act = ac_mod.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, rew, terminated, truncated, _ = env.step(act)
                ep_len, ep_ret = ep_len + 1, ep_ret + rew
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    for env_id in range(env.num_envs):
                        if done[env_id]:
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret[env_id])
                            ep_len[env_id], ep_ret[env_id] = 0, 0
            obs, _ = env.reset()

            if (epoch % save_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
            if ((epoch + 1) % checkpoint_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, f'model{epoch+1}.pt'))

            # Log info about epoch
            ep_lens, ep_rets, q_vals = np.array(ep_lens), np.array(ep_rets), np.array(q_vals)
            writer.add_scalar('EpLen/mean', ep_lens.mean(), (epoch+1)*steps_per_epoch)
            writer.add_scalar('EpRet/mean', ep_rets.mean(), (epoch+1)*steps_per_epoch)
            writer.add_scalar('EpRet/max', ep_rets.max(), (epoch+1)*steps_per_epoch)
            writer.add_scalar('EpRet/min', ep_rets.min(), (epoch+1)*steps_per_epoch)
            writer.add_scalar('QVals/mean', q_vals.mean(), epoch+1)
            writer.add_scalar('QVals/max', q_vals.max(), epoch+1)
            writer.add_scalar('QVals/min', q_vals.min(), epoch+1)
            writer.add_scalar('Time', time.time()-start_time, epoch+1)
            writer.flush()
            q_vals = []
        
        # Save final model
        torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
        writer.close()
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--hid_act', type=int, default=64)
    parser.add_argument('--hid_cri', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--action_std', type=float, default=0.1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--q_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Set directory for logging
    log_dir = os.getcwd() + '/../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Actor-Critic network kwargs
    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l,
                     hidden_sizes_critic=[args.hid_cri]*args.l,
                     hidden_acts_actor=torch.nn.ReLU, 
                     hidden_acts_critic=torch.nn.ReLU,
                     action_std=args.action_std)
    
    # Setup lambda for initializing asynchronous vectorized environments
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    env_fn = [lambda: gym.make(args.env, max_episode_steps=max_ep_len)] * args.cpu

    # Begin training
    trainer = TD3Trainer()
    trainer.train_mod(env_fn, model_path=args.model_path, ac=MLPActorCritic, 
                      ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                      epochs=args.epochs, buf_size=args.buf_size, gamma=args.gamma, 
                      polyak=args.polyak, pi_lr=args.pi_lr, q_lr=args.q_lr, 
                      batch_size=args.batch_size, start_steps=args.start_steps, 
                      learning_starts=args.learning_starts, update_every=args.update_every, 
                      target_noise=args.target_noise, noise_clip=args.noise_clip, 
                      policy_delay=args.policy_delay, num_test_episodes=args.num_test_episodes, 
                      log_dir=log_dir, save_freq=args.save_freq, 
                      checkpoint_freq=args.checkpoint_freq)