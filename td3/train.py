import os
import gymnasium as gym
from gymnasium.spaces import Box
import time
import numpy as np
import torch
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, mpi_fork, num_procs
from spinup.utils.run_utils import setup_logger_kwargs

from models import MLPActorCritic, polyak_average

class ReplayBuffer:
    def __init__(self, env: gym.Env, buf_size, batch_size):
        # Check the type of the action space
        if not (isinstance(env.action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.ctr, self.buf_full = 0, False
        self.buf_size, self.batch_size = buf_size, batch_size
        obs_shape, act_shape = env.observation_space.shape, env.action_space.shape
        buf_shape = tuple([buf_size])

        # Initialize all buffers for storing data during training
        self.obs = np.zeros(buf_shape + obs_shape, dtype=np.float32)
        self.act = np.zeros(buf_shape + act_shape, dtype=np.float32)
        self.rew = np.zeros((buf_size,), dtype=np.float32)
        self.q_val = np.zeros((buf_size,), dtype=np.float32)
        self.done = np.zeros((buf_size,), dtype=np.bool)

    def update_buffer(self, obs, act, rew, q_val, done):
        self.obs[self.ctr] = obs
        self.act[self.ctr] = act
        self.rew[self.ctr] = rew
        self.q_val[self.ctr] = q_val
        self.done[self.ctr] = done

        # Update buffer counter and reset if neccessary
        self.ctr += 1
        if self.ctr == self.buf_size:
            self.ctr = 0
            self.buf_full = True

    def get_batch(self):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype)

        # Generate random indices
        size = self.buf_size if self.buf_full==True else self.ctr
        indices = np.random.choice(size-1, self.batch_size, replace=False)

        # Return randomly selected experience tuples
        return to_tensor(self.obs[indices], torch.float32), to_tensor(self.act[indices], torch.float32), \
                to_tensor(self.rew[indices], torch.float32), to_tensor(self.obs[indices+1], torch.float32), \
                to_tensor(self.done[indices], torch.bool)
    
    
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
                        q1_optim: Adam, q2_optim: Adam, logger: EpochLogger, gamma, 
                        polyak, target_noise, noise_clip, action_max, update_policy=False):
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
        mpi_avg_grads(ac.critic_1.net)
        mpi_avg_grads(ac.critic_2.net)
        q1_optim.step()
        q2_optim.step()

        # Update target critic networks
        polyak_average(ac.critic_1.net.parameters(), ac.critic_1.net_target.parameters(), polyak)
        polyak_average(ac.critic_2.net.parameters(), ac.critic_2.net_target.parameters(), polyak)

        # Log training statistics
        logger.store(LossQ1=loss_q1.item(), LossQ2=loss_q2.item())

        if update_policy == True:
            # Update the actor network (critic's weights are frozen temporarily)
            ac.critic_1.set_grad_tracking(val=False)
            pi_optim.zero_grad()
            loss_pi = self.__calc_pi_loss(ac, obs)
            loss_pi.backward()
            mpi_avg_grads(ac.actor.net)
            pi_optim.step()
            ac.critic_1.set_grad_tracking(val=True)

            # Update target network
            polyak_average(ac.actor.net.parameters(), ac.actor.net_target.parameters(), polyak)

            # Log training statistics
            logger.store(LossPi=loss_pi.item())
    
    def train_mod(self, env_fn, model_path='', ac=MLPActorCritic, ac_kwargs=dict(), 
                  seed=0, steps_per_epoch=4000, epochs=100, buf_size=1000000, 
                  gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
                  batch_size=100, start_steps=10000, learning_starts=1000,
                  update_every=50, target_noise=0.2, noise_clip=0.5, 
                  policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
                  logger_kwargs=dict(), save_freq=10, checkpoint_freq=20):
        setup_pytorch_for_mpi()

        # Initialize logger
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals()) 
        
        local_buf_size = buf_size // num_procs()
        local_steps_per_epoch = steps_per_epoch // num_procs()
        local_start_steps = start_steps // num_procs()
        local_learning_starts = learning_starts // num_procs()

        # Setup random seed number for PyTorch and NumPy
        seed += 10000 * proc_id()
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # Initialize environment and Actor-Critic
        env = env_fn()
        if len(model_path) > 0:
            ac_mod = torch.load(model_path)
        else:
            ac_mod = ac(env, **ac_kwargs)

        # Initialize the experience replay buffer for training
        buf = ReplayBuffer(env, local_buf_size, batch_size)    

        # Store action limits for target action clipping
        action_max = torch.tensor(env.action_space.high) 

        # Sync ac network parameters and initialize optimizers
        sync_params(ac_mod)
        pi_optim = Adam(ac_mod.actor.net.parameters(), lr=pi_lr)
        q1_optim = Adam(ac_mod.critic_1.net.parameters(), lr=q_lr)
        q2_optim = Adam(ac_mod.critic_2.net.parameters(), lr=q_lr)
        logger.setup_pytorch_saver(ac_mod)

        # Initialize environment variables
        obs, _ = env.reset(seed=seed)
        ep_len, start_time = 0, time.time()

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                if (step + local_steps_per_epoch*epoch) > local_start_steps:
                    act, q_val = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    act = env.action_space.sample()
                    _, q_val = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                    
                obs_next, rew, terminated, truncated, _ = env.step(act)

                buf.update_buffer(obs, act, rew, q_val, terminated)
                logger.store(QVals=q_val)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_steps_per_epoch-1)
                truncated = (ep_len == max_ep_len) or truncated

                if epoch_done or terminated or truncated:
                    obs, _, ep_len = env.reset(), 0
                
                if (buf.buf_full or (buf.ctr > local_learning_starts)) \
                    and ((step % update_every) == 0):
                    for j in range(update_every):
                        update_policy = (j % policy_delay) == 0 
                        self.__update_params(ac_mod, buf, pi_optim, q1_optim, 
                                             q2_optim, logger, gamma, polyak, 
                                             target_noise, noise_clip, action_max,
                                             update_policy=update_policy)
            
            # Evaluate deterministic policy
            for _ in range(num_test_episodes):
                ep_len, ep_ret = 0, 0
                obs, _, done = env.reset(), False
                while not done:
                    act = ac_mod.act(torch.as_tensor(obs, dtype=torch.float32))
                    obs, rew, terminated, truncated, _ = env.step(act)
                    ep_len, ep_ret = ep_len + 1, ep_ret + rew
                    done = terminated or truncated
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, _ = env.reset()

            if (epoch % save_freq) == 0:
                logger.save_state({'env': env})
            if ((epoch + 1) % checkpoint_freq) == 0:
                logger.save_state({'env': env}, itr=epoch+1)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        
        # Save final model
        logger.save_state({'env': env})
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
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='td3_custom')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Launch MPI processes
    mpi_fork(args.cpu) 

    # Actor-Critic network kwargs
    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l,
                     hidden_sizes_critic=[args.hid_cri]*args.l,
                     hidden_acts_actor=torch.nn.ReLU, 
                     hidden_acts_critic=torch.nn.ReLU,
                     action_std=args.action_std)
    
    # EpochLogger kwargs
    data_dir = os.getcwd() + '/../data/td3/' + args.env + '/'
    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=data_dir, seed=args.seed)

    # Begin training
    trainer = TD3Trainer()
    trainer.train_mod(lambda : gym.make(args.env), model_path=args.model_path, ac=MLPActorCritic, 
                      ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                      epochs=args.epochs, buf_size=args.buf_size, gamma=args.gamma, 
                      polyak=args.polyak, pi_lr=args.pi_lr, q_lr=args.q_lr, 
                      batch_size=args.batch_size, start_steps=args.start_steps, 
                      learning_starts=args.learning_starts, update_every=args.update_every, 
                      target_noise=args.target_noise, noise_clip=args.noise_clip, 
                      policy_delay=args.policy_delay, num_test_episodes=args.num_test_episodes, 
                      max_ep_len=args.max_ep_len, logger_kwargs=logger_kwargs, 
                      save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)