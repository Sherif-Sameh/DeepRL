import gym
from gym.spaces import Box
import time
import numpy as np
import torch
from torch import nn
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
        self.logp = np.zeros((buf_size,), dtype=np.float32)
        self.rew = np.zeros((buf_size,), dtype=np.float32)
        self.q_val = np.zeros((buf_size,), dtype=np.float32)
        self.done = np.zeros((buf_size,), dtype=np.bool)

    def update_buffer(self, obs, act, logp, rew, q_val, done):
        self.obs[self.ctr] = obs
        self.logp[self.ctr] = logp
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
                to_tensor(self.logp[indices], torch.float32), to_tensor(self.rew[indices], torch.float32), \
                to_tensor(self.obs[indices+1], torch.float32), to_tensor(self.done[indices], torch.bool)
    

class AlphaModule(nn.Module):
    def __init__(self, alpha_init=1.0, requires_grad=True):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([alpha_init], dtype=torch.float32)), 
                                      requires_grad=requires_grad)

    def forward(self):
        return torch.exp(self.log_alpha)
    

class SACTrainer:
    def __calc_q_loss(self, ac: MLPActorCritic, gamma, alpha, obs: torch.Tensor, act: torch.Tensor, 
                      rew: torch.Tensor, obs_next: torch.Tensor, done: torch.Tensor):
        # Get current Q values from both Q networks
        q_vals_1 = ac.critic_1.forward(obs, act)
        q_vals_2 = ac.critic_2.forward(obs, act)
        q_vals_1.squeeze_(dim=1)
        q_vals_2.squeeze_(dim=1)
        
        # Get actions and their log probs for next observations
        act_target, log_prob_target = ac.actor.log_prob_no_grad(obs_next)

        # Determine target Q values and mask values where done is True
        q_vals_target_1 = ac.critic_1.forward_target(obs_next, act_target)
        q_vals_target_2 = ac.critic_2.forward_target(obs_next, act_target)
        q_vals_target = torch.min(q_vals_target_1, q_vals_target_2)
        q_vals_target.squeeze_(dim=1)
        q_vals_target -= alpha * log_prob_target
        q_vals_target[done] = 0.0

        # Calculate TD target and errors
        td_target = rew + gamma * q_vals_target
        td_error_1 = td_target - q_vals_1
        td_error_2 = td_target - q_vals_2

        return (td_error_1**2).mean(), (td_error_2**2).mean()
    
    def __calc_pi_loss(self, ac: MLPActorCritic, alpha, obs: torch.Tensor):
        # Re-propagate actions and their log probs through actor for gradient tracking
        act, log_prob = ac.actor.log_prob(obs)  

        # Evaluate the minimum of the two critics' Q values
        q_vals_1 = ac.critic_1.forward(obs, act)
        q_vals_2 = ac.critic_2.forward(obs, act)
        q_vals = torch.min(q_vals_1, q_vals_2)
        q_vals.squeeze_(dim=1)

        return (-q_vals + alpha * log_prob).mean()

    def __calc_alpha_loss(self, alpha_mod: AlphaModule, entropy_target, logp: torch.Tensor):
        alpha = alpha_mod.forward()
        
        return (alpha * (-logp - entropy_target)).mean()
    
    def __update_params(self, ac: MLPActorCritic, alpha_mod: AlphaModule, buf: ReplayBuffer, 
                        pi_optim: Adam, q1_optim: Adam, q2_optim: Adam, alpha_optim: Adam, 
                        logger: EpochLogger, gamma, polyak, entropy_target, update_alpha=True):
        # Get mini-batch from replay buffer
        obs, act, logp, rew, obs_next, done = buf.get_batch()
        alpha_det = alpha_mod.forward().detach()
        
        # Update critic networks
        q1_optim.zero_grad()
        q2_optim.zero_grad()
        loss_q1, loss_q2 = self.__calc_q_loss(ac, gamma, alpha_det, obs, 
                                              act, rew, obs_next, done)
        loss_q1.backward()
        loss_q2.backward()
        mpi_avg_grads(ac.critic_1.net)
        mpi_avg_grads(ac.critic_2.net)
        q1_optim.step()
        q2_optim.step()

        # Update the actor network (critics' weights are frozen temporarily)
        ac.critic_1.set_grad_tracking(val=False)
        ac.critic_2.set_grad_tracking(val=False)
        pi_optim.zero_grad()
        loss_pi = self.__calc_pi_loss(ac, alpha_det, obs)
        loss_pi.backward()
        mpi_avg_grads(ac.actor.net)
        pi_optim.step()
        ac.critic_1.set_grad_tracking(val=True)
        ac.critic_2.set_grad_tracking(val=True)

        # Update the log alpha parameter if its estimated online 
        if update_alpha == True:
            alpha_optim.zero_grad()
            loss_alpha = self.__calc_alpha_loss(alpha_mod, entropy_target, logp)
            loss_alpha.backward()
            mpi_avg_grads(alpha_mod)
            alpha_optim.step()
            logger.store(LossAlpha=loss_alpha.item())

        # Update target critic networks
        polyak_average(ac.critic_1.net.parameters(), ac.critic_1.net_target.parameters(), polyak)
        polyak_average(ac.critic_2.net.parameters(), ac.critic_2.net_target.parameters(), polyak)

        # Log training statistics
        logger.store(LossQ1=loss_q1.item(), LossQ2=loss_q2.item(), 
                     LossPi=loss_pi.item())

    def train_mod(self, env_fn, model_path='', ac=MLPActorCritic, ac_kwargs=dict(), 
                  seed=0, steps_per_epoch=4000, epochs=100, buf_size=1000000, 
                  gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, 
                  entropy_target=None, auto_alpha=False, batch_size=100, 
                  start_steps=10000, learning_starts=1000, update_every=50, 
                  num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(), 
                  save_freq=10, checkpoint_freq=20):
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

        # Initialize log alpha module based on training configuration
        if auto_alpha == True:
            alpha_init = alpha if alpha is not None else 1.0
            entropy_target = -np.prod(env.action_space.shape, dtype=np.float32) \
                if entropy_target is None else entropy_target
            alpha_mod = AlphaModule(alpha_init=alpha_init, requires_grad=True)
            sync_params(alpha_mod)
        else:
            entropy_target = None
            alpha_mod = AlphaModule(alpha_init=alpha, requires_grad=False)

        # Sync ac network parameters and initialize optimizers
        sync_params(ac_mod)
        pi_optim = Adam(ac_mod.actor.net.parameters(), lr=lr)
        q1_optim = Adam(ac_mod.critic_1.net.parameters(), lr=lr)
        q2_optim = Adam(ac_mod.critic_2.net.parameters(), lr=lr)
        alpha_optim = Adam(alpha_mod.parameters(), lr=lr)
        logger.setup_pytorch_saver(ac_mod)

        # Initialize environment variables
        obs, ep_len = env.reset(), 0
        start_time = time.time()

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                if (step + local_steps_per_epoch*epoch) > local_start_steps:
                    act, q_val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    act = env.action_space.sample()
                    _, q_val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                    
                obs_next, rew, done, _ = env.step(act)

                buf.update_buffer(obs, act, logp, rew, q_val, done)
                logger.store(QVals=q_val)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_steps_per_epoch-1)
                max_ep_len_reached = ep_len == max_ep_len

                if epoch_done or done or max_ep_len_reached:
                    obs, ep_len = env.reset(), 0
                
                if (buf.buf_full or (buf.ctr > local_learning_starts)) \
                    and ((step % update_every) == 0):
                    for _ in range(update_every):
                        self.__update_params(ac_mod, alpha_mod, buf, pi_optim, 
                                             q1_optim, q2_optim, alpha_optim, 
                                             logger, gamma, polyak, entropy_target, 
                                             update_alpha=auto_alpha)
            
            # Evaluate deterministic policy
            for _ in range(num_test_episodes):
                ep_len, ep_ret = 0, 0
                obs, done = env.reset(), False
                while not done:
                    act = ac_mod.act(torch.as_tensor(obs, dtype=torch.float32))
                    obs, rew, done, _ = env.step(act)
                    ep_len, ep_ret = ep_len + 1, ep_ret + rew
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs = env.reset()

            if (epoch % save_freq) == 0:
                logger.save_state({'env': env})
            if ((epoch + 1) % checkpoint_freq) == 0:
                logger.save_state({'env': env}, itr=epoch+1)
            
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('Alpha', alpha_mod.forward().item())
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            if auto_alpha == True:
                logger.log_tabular('LossAlpha', average_only=True)
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
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto_alpha', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='sac_custom')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Launch MPI processes
    mpi_fork(args.cpu) 

    # Actor-Critic network kwargs
    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l,
                     hidden_sizes_critic=[args.hid_cri]*args.l,
                     hidden_acts_actor=torch.nn.ReLU, 
                     hidden_acts_critic=torch.nn.ReLU)
    
    # EpochLogger kwargs
    data_dir = '/home/sherif/user/python/DeepRL/data/sac'
    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=data_dir, seed=args.seed)

    # Begin training
    trainer = SACTrainer()
    trainer.train_mod(lambda : gym.make(args.env), model_path=args.model_path, ac=MLPActorCritic, 
                      ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                      epochs=args.epochs, buf_size=args.buf_size, gamma=args.gamma, 
                      polyak=args.polyak, lr=args.lr, alpha=args.alpha, 
                      auto_alpha=args.auto_alpha, batch_size=args.batch_size, 
                      start_steps=args.start_steps, learning_starts=args.learning_starts, 
                      update_every=args.update_every, num_test_episodes=args.num_test_episodes, 
                      max_ep_len=args.max_ep_len, logger_kwargs=logger_kwargs, 
                      save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)