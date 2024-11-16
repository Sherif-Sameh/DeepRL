import gym
from gym.spaces import Discrete, Box
import time
import numpy as np
import torch
from torch import nn
import torch.distributions
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, mpi_fork, mpi_statistics_scalar, num_procs, mpi_avg
from spinup.utils.run_utils import setup_logger_kwargs

from models import MLPActorCritic, MLPActor, MLPCritic

class PPOBuffer:
    def __init__(self, env: gym.Env, buf_size, batch_size, gamma, lam):
        # Check the type of the action space
        if not (isinstance(env.action_space, Discrete) or 
                isinstance(env.action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.gamma, self.lam = gamma, lam
        self.ep_start, self.batch_size = 0, batch_size
        obs_shape, act_shape = env.observation_space.shape, env.action_space.shape
        buf_shape = tuple([buf_size])

        # Initialize all buffers for storing data during an epoch and training
        self.obs = np.zeros(buf_shape + obs_shape, dtype=np.float32)
        self.act = np.zeros(buf_shape + act_shape, dtype=np.float32)
        self.rew = np.zeros((buf_size,), dtype=np.float32)
        self.rtg = np.zeros((buf_size,), dtype=np.float32)
        self.adv = np.zeros((buf_size,), dtype=np.float32)
        self.val = np.zeros((buf_size,), dtype=np.float32)
        self.logp = np.zeros((buf_size,), dtype=np.float32)

    def update_buffer(self, obs, act, rew, val, logp, ep_len):
        self.obs[self.ep_start + ep_len] = obs
        self.act[self.ep_start + ep_len] = act
        self.rew[self.ep_start + ep_len] = rew
        self.val[self.ep_start + ep_len] = val
        self.logp[self.ep_start + ep_len] = logp

    def terminate_ep(self, ep_length, val_terminal=0, epoch_done=True):
        # Calculate per episode statistics - Return to Go 
        ep_end = self.ep_start + ep_length
        self.rtg[ep_end-1] = self.rew[ep_end-1] + self.gamma*val_terminal
        for i in range(ep_length-2, -1, -1):
            self.rtg[self.ep_start+i] = self.rew[self.ep_start+i] + self.gamma * self.rtg[self.ep_start+i+1]
                                               
        # Calculate per episode statistics - Advantage function
        ep_slice = slice(self.ep_start, ep_end)
        rews = self.rew[ep_slice]
        vals = np.append(self.val[ep_slice], val_terminal)
        deltas = rews + self.gamma * vals[1:] - vals[:-1]
        self.adv[ep_end-1] = deltas[-1]
        for i in range(ep_length-2, -1, -1):
            self.adv[self.ep_start+i] = deltas[i] + self.gamma * self.lam * self.adv[self.ep_start+i+1]

        # Update new episode start index
        self.ep_start = 0 if epoch_done==True else self.ep_start + ep_length

    def get_pi_dataloader(self):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32)
        dataset = TensorDataset(to_tensor(self.obs), to_tensor(self.act), 
                                to_tensor(self.adv), to_tensor(self.logp))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32)
        dataset = TensorDataset(to_tensor(self.obs), to_tensor(self.rtg))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            

class PPOTrainer:
    def __calc_policy_loss(self, actor: MLPActor, obs: torch.Tensor, act: torch.Tensor,
                           adv: torch.Tensor, logp: torch.Tensor, clip_ratio):
        log_prob = actor.log_prob_grad(obs, act)
        ratio = torch.exp(log_prob - logp)
        clipped_ratio = torch.clamp(ratio, 
                                    1 - clip_ratio,
                                    1 + clip_ratio)
        surrogate_obj = -(torch.min(ratio * adv, other=(clipped_ratio * adv))).mean()
        
        return surrogate_obj
    
    def __calc_val_loss(self, critic: MLPCritic, obs: torch.Tensor, rtg: torch.Tensor):
        val = critic.forward_grad(obs)

        return ((val - rtg)**2).mean()
    
    def __update_params(self, ac_mod: MLPActorCritic, buf: PPOBuffer, 
                        pi_optim: Adam, val_optim: Adam, logger: EpochLogger,
                        train_pi_iters, train_v_iters, clip_ratio, target_kl):
        # Store old policy for KL early stopping
        ac_mod.actor.update_policy(torch.as_tensor(buf.obs, dtype=torch.float32))
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
                # Normalize advantages mini-batch wise across all MPI processes
                adv_mean, adv_std = mpi_statistics_scalar(adv.numpy())
                adv = (adv - adv_mean) / adv_std

                pi_optim.zero_grad()
                loss_pi = self.__calc_policy_loss(ac_mod.actor, obs, act,
                                                  adv, logp, clip_ratio)
                loss_pi.backward()
                mpi_avg_grads(ac_mod.actor)
                pi_optim.step()
            
            # Check KL-Divergence constraint for triggering early stopping
            kl = ac_mod.actor.kl_divergence(torch.as_tensor(buf.obs, dtype=torch.float32), 
                                            pi_curr)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                print(f'Actor updates cut-off after {i+1} iterations by KL {kl}')
                break

        # Loop train_v_iters times over the whole dataset to update value function
        loss_val_old = self.__calc_val_loss(ac_mod.critic, 
                                            torch.as_tensor(buf.obs, dtype=torch.float32), 
                                            torch.as_tensor(buf.rtg, dtype=torch.float32))
        for i in range(train_v_iters):
            # Get dataloader for performing mini-batch SGD updates
            dataloader = buf.get_val_dataloader()
            
            # Loop over dataset in mini-batches
            for obs, rtg in dataloader:
                val_optim.zero_grad()
                loss_val = self.__calc_val_loss(ac_mod.critic, obs, rtg)
                loss_val.backward()
                mpi_avg_grads(ac_mod.critic)
                val_optim.step()

        # Log epoch statistics
        logger.store(LossPi=loss_pi.item(), LossV=loss_val.item(),
                     KL=kl.item(),
                     DeltaLossV=(loss_val - loss_val_old).item())
    
    def train_mod(self, env_fn, ac=MLPActorCritic, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=4096, batch_size=512, epochs=50, gamma=0.99, 
                  clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, 
                  train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, 
                  logger_kwargs=dict(), save_freq=10):
        setup_pytorch_for_mpi()
        local_steps_per_epoch = steps_per_epoch // num_procs()
        local_batch_size = batch_size // num_procs()

        # Setup random seed number for PyTorch and NumPy
        seed += 10000 * proc_id()
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # Initialize environment, actor-critic and logger
        env = env_fn()
        ac_mod = ac(env, **ac_kwargs)
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals()) 

        # Initialize the experience buffer for training
        buf_mod = PPOBuffer(env, local_steps_per_epoch, local_batch_size,
                            gamma, lam)

        # Sync AC parameters and initialize optimizers
        sync_params(ac_mod)
        pi_optim = Adam(ac_mod.actor.parameters(), lr=pi_lr)
        val_optim = Adam(ac_mod.critic.parameters(), lr=vf_lr)
        logger.setup_pytorch_saver(ac_mod)

        # Initialize environment variables
        obs = env.reset()
        ep_len, ep_ret = 0, 0
        start_time = time.time()

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                act, val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                obs_next, rew, done, _ = env.step(act)

                buf_mod.update_buffer(obs, act, rew, val, logp, ep_len)
                logger.store(VVals=val)
                obs, ep_len, ep_ret = obs_next, ep_len + 1, ep_ret + rew

                epoch_done = step == (local_steps_per_epoch-1)
                max_ep_len_reached = ep_len == max_ep_len
                terminal = done or max_ep_len_reached

                if epoch_done or terminal:
                    val_terminal = 0 if done else ac_mod.critic(torch.as_tensor(obs, dtype=torch.float32))
                    buf_mod.terminate_ep(ep_len, val_terminal=val_terminal, epoch_done=epoch_done)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs = env.reset()
                    ep_len, ep_ret = 0, 0
            
            if (epoch % save_freq) == 0:
                logger.save_state({'env': env})
                
            self.__update_params(ac_mod, buf_mod, pi_optim, val_optim, logger,
                                 train_pi_iters, train_v_iters, clip_ratio, target_kl)
            
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        
        # Save final model
        logger.save_state({'env': env})
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid_act', type=int, default=64)
    parser.add_argument('--hid_cri', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo_custom')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Launch MPI processes
    mpi_fork(args.cpu) 

    # Actor-Critic kwargs
    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l,
                     hidden_sizes_critic=[args.hid_cri]*args.l,
                     hidden_acts_actor=torch.nn.Tanh,
                     hidden_acts_critic=torch.nn.Tanh)
    
    # EpochLogger kwargs
    data_dir = '/home/sherif/user/python/DRL/data/ppo'
    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=data_dir)

    # Begin training
    trainer = PPOTrainer()
    trainer.train_mod(lambda : gym.make(args.env), ac=MLPActorCritic, ac_kwargs=ac_kwargs,
                      seed=args.seed, steps_per_epoch=args.steps, batch_size=args.batch_size,
                      epochs=args.epochs, gamma=args.gamma, clip_ratio=args.clip_ratio,
                      pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters,
                      train_v_iters=args.train_v_iters, lam=args.lam, max_ep_len=args.max_ep_len, 
                      logger_kwargs=logger_kwargs, save_freq=args.save_freq)