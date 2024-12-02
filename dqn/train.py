import os
import gymnasium as gym
from gymnasium.spaces import Discrete
import time
import numpy as np
import torch
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, mpi_fork, num_procs
from spinup.utils.run_utils import setup_logger_kwargs

from models import MLPDQN, MLPDuelingDQN, MLPDDQN, MLPDuelingDDQN

class ReplayBuffer:
    def __init__(self, env: gym.Env, buf_size, batch_size):
        # Check the type of the action space
        if not (isinstance(env.action_space, Discrete)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.ctr, self.buf_full = 0, False
        self.buf_size, self.batch_size = buf_size, batch_size
        obs_shape, act_shape = env.observation_space.shape, env.action_space.shape
        buf_shape = tuple([buf_size])

        # Initialize all buffers for storing data during training
        self.obs = np.zeros(buf_shape + obs_shape, dtype=np.float32)
        self.act = np.zeros(buf_shape + act_shape, dtype=np.int64)
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
        return to_tensor(self.obs[indices], torch.float32), to_tensor(self.act[indices], torch.int64), \
                to_tensor(self.rew[indices], torch.float32), to_tensor(self.obs[indices+1], torch.float32), \
                to_tensor(self.done[indices], torch.bool)

    def get_weights(self):
        return torch.ones(self.batch_size, dtype=torch.float32)
    
    def update_beta(self):
        return
    
    def update_priorities(self, td_errors):
        return

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, env: gym.Env, buf_size, batch_size, alpha, 
                 beta0, beta_rate, epsilon):
        super().__init__(env, buf_size, batch_size)

        # Store prioritized replay buffer parameters
        self.alpha = alpha
        self.beta = beta0
        self.beta_rate = beta_rate
        self.epsilon = epsilon

        # Initialize transition priority buffer and maximum priority value
        self.pri = np.zeros((buf_size,), dtype=np.float32)
        self.pri_max = 1.0

        # Store weigths and indices for each sample in a mini-batch
        self.weights = np.ones((self.batch_size,), dtype=np.float32)
        self.indices = np.zeros((self.batch_size,), dtype=np.int64)

    def update_buffer(self, obs, act, rew, q_val, done):
        # Normal replay buffer updates
        self.obs[self.ctr] = obs
        self.act[self.ctr] = act
        self.rew[self.ctr] = rew
        self.q_val[self.ctr] = q_val
        self.done[self.ctr] = done

        # Set priorities of new transitions to max priority
        self.pri[self.ctr] = self.pri_max

        # Update buffer counter and reset if neccessary
        self.ctr += 1
        if self.ctr == self.buf_size:
            self.ctr = 0
            self.buf_full = True

    def get_batch(self):
        to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype)

        # Calculate sampling proabilities for each transition
        size = self.buf_size if self.buf_full==True else self.ctr
        prob = np.zeros((size-1,), dtype=np.float32)
        prob = self.pri[:size-1]**self.alpha
        prob = prob / np.sum(prob)
        
        # Sample weighted random indices and update weights for parameter updates
        self.indices = np.random.choice(size-1, self.batch_size, replace=False, p=prob)
        self.weights = (self.buf_size * prob[self.indices])**(-self.beta)
        self.weights = self.weights / np.max(self.weights)

        # Return randomly selected experience tuples
        return to_tensor(self.obs[self.indices], torch.float32), to_tensor(self.act[self.indices], torch.int64), \
                to_tensor(self.rew[self.indices], torch.float32), to_tensor(self.obs[self.indices+1], torch.float32), \
                to_tensor(self.done[self.indices], torch.bool)

    def get_weights(self):
        return torch.as_tensor(self.weights, dtype=torch.float32)

    def update_beta(self):
        self.beta = min(self.beta + self.beta_rate, 1.0)

    def update_priorities(self, td_errors):
        self.pri[self.indices] = np.abs(td_errors) + self.epsilon
        self.pri_max = max(self.pri_max, np.max(self.pri[self.indices]))

    
class DQNTrainer:
    def __calc_td_error(self, q_net: MLPDQN, gamma, obs: torch.Tensor, act: torch.Tensor,
                      rew: torch.Tensor, obs_next: torch.Tensor, done: torch.Tensor):
        # Get current and target Q vals and mask target Q vals where done is True
        q_vals = q_net.forward_grad(obs, act)
        q_vals_target = q_net.forward_target(obs_next)
        q_vals_target[done] = 0.0

        # Calculate TD target and error
        td_target = rew + gamma * q_vals_target
        td_error = td_target - q_vals

        return td_error
    
    def __update_params(self, q_net: MLPDQN, buf: ReplayBuffer, 
                        q_optim: Adam, logger: EpochLogger, 
                        gamma, update_target_network):
        # Get mini-batch and perform one parameter update for the Q network
        obs, act, rew, obs_next, done = buf.get_batch()
        q_optim.zero_grad()
        td_error = self.__calc_td_error(q_net, gamma, obs, act, 
                                    rew, obs_next, done)
        loss_weights = buf.get_weights()
        loss_q = ((td_error * loss_weights)**2).mean()
        loss_q.backward()
        mpi_avg_grads(q_net.net)
        q_optim.step()
        buf.update_priorities(td_error.detach().numpy())
        td_error = self.__calc_td_error(q_net, gamma, obs, act, 
                                        rew, obs_next, done)
        loss_q_new = ((td_error * loss_weights)**2).mean()

        # Update target Q network values if neccessary
        if update_target_network == True:
            q_net.update_target()

        # Log epoch statistics
        logger.store(LossQ=loss_q.item(),
                     DeltaLossQ=(loss_q_new - loss_q).item())
    
    def train_mod(self, env_fn, model_path='', dueling=True, double_q=True, 
                  q_net_kwargs=dict(), seed=0, prioritized_replay=False, 
                  prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, 
                  prioritized_replay_beta_rate=None, prioritized_replay_eps=1e-6, 
                  eps_init=1.0, eps_final=0.05, eps_decay_rate=0.2, buf_size=10000, 
                  steps_per_epoch=4000, batch_size=32, epochs=100, max_ep_len=1000, 
                  learning_starts=1000, train_freq=1, target_network_update_freq=500, 
                  gamma=0.99, q_lr=5e-4, logger_kwargs=dict(), save_freq=10,
                  checkpoint_freq=20):
        setup_pytorch_for_mpi()

        # Initialize logger
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals()) 

        local_buf_size = buf_size // num_procs()
        local_steps_per_epoch = steps_per_epoch // num_procs()
        local_learning_starts = learning_starts // num_procs()

        # Setup random seed number for PyTorch and NumPy
        seed += 10000 * proc_id()
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # Determine DQN variant to use
        if (dueling==True) and (double_q==True):
            q_net = MLPDuelingDDQN
        elif dueling==True:
            q_net = MLPDuelingDQN
        elif double_q==True:
            q_net = MLPDDQN
        else:
            q_net = MLPDQN 

        # Initialize environment and Q network
        env = env_fn()
        if len(model_path) > 0:
            q_net_mod = torch.load(model_path)
        else:
            q_net_mod = q_net(env, eps_init, eps_final, eps_decay_rate, **q_net_kwargs)

        # Initialize the experience replay buffer for training
        if prioritized_replay==True:
            prioritized_replay_beta_rate = (1.0 - prioritized_replay_beta0)/(epochs-1) \
                if prioritized_replay_beta_rate is None else prioritized_replay_beta_rate
            buf = PrioritizedReplayBuffer(env, local_buf_size, batch_size, 
                                          prioritized_replay_alpha, prioritized_replay_beta0,
                                          prioritized_replay_beta_rate, prioritized_replay_eps)
        else:
            buf = ReplayBuffer(env, local_buf_size, batch_size)

        # Sync Q network parameters and initialize optimizers
        sync_params(q_net_mod)
        q_optim = Adam(q_net_mod.parameters(), lr=q_lr)
        logger.setup_pytorch_saver(q_net_mod)

        # Initialize environment variables
        obs, _ = env.reset(seed=seed)
        ep_len, start_time = 0, time.time()

        for epoch in range(epochs):
            for step in range(local_steps_per_epoch):
                act, q_val = q_net_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                obs_next, rew, terminated, truncated, _ = env.step(act)

                buf.update_buffer(obs, act, rew, q_val, terminated)
                logger.store(QVals=q_val)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_steps_per_epoch-1)
                truncated = (ep_len == max_ep_len) or truncated

                if epoch_done or terminated or truncated:
                    obs, _, ep_len = env.reset(), 0
                
                if (buf.buf_full or (buf.ctr > local_learning_starts)) \
                    and ((step % train_freq) == 0):
                    update_target_network = (step % target_network_update_freq) == 0
                    self.__update_params(q_net_mod, buf, q_optim, logger, 
                                         gamma, update_target_network)
            
            # Evaluate deterministic policy
            for _ in range(10):
                ep_len, ep_ret = 0, 0
                obs, _, done = env.reset(), False
                while not done:
                    act = q_net_mod.act(torch.as_tensor(obs, dtype=torch.float32))
                    obs, rew, terminated, truncated, _ = env.step(act)
                    ep_len, ep_ret = ep_len + 1, ep_ret + rew
                    done = terminated or truncated
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, _ = env.reset()

            if (epoch % save_freq) == 0:
                logger.save_state({'env': env})
            if ((epoch + 1) % checkpoint_freq) == 0:
                logger.save_state({'env': env}, itr=epoch+1)

            q_net_mod.update_eps_exp()
            buf.update_beta()
            
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('DeltaLossQ', average_only=True)
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
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--dueling', type=bool, default=True)
    parser.add_argument('--double_q', type=bool, default=True)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--prioritized_replay', type=bool, default=False)
    parser.add_argument('--eps_init', type=float, default=1.0)
    parser.add_argument('--eps_final', type=float, default=0.1)
    parser.add_argument('--eps_decay_rate', type=float, default=0.2)
    parser.add_argument('--buf_size', type=int, default=1000000)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--target_network_update_freq', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--q_lr', type=float, default=5e-4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='dqn_custom')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Launch MPI processes
    mpi_fork(args.cpu) 

    # Q network kwargs
    q_net_kwargs = dict(hidden_sizes=[args.hid]*args.l,
                        hidden_acts=torch.nn.ReLU)
    
    # EpochLogger kwargs
    data_dir = os.getcwd() + '/../data/dqn/' + args.env + '/'
    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=data_dir, seed=args.seed)

    # Begin training
    trainer = DQNTrainer()
    trainer.train_mod(lambda : gym.make(args.env), model_path=args.model_path, dueling=args.dueling, 
                      double_q=args.double_q, q_net_kwargs=q_net_kwargs, 
                      prioritized_replay=args.prioritized_replay, seed=args.seed, 
                      eps_init=args.eps_init, eps_final=args.eps_final, 
                      eps_decay_rate=args.eps_decay_rate, buf_size=args.buf_size, 
                      steps_per_epoch=args.steps, batch_size=args.batch_size,
                      epochs=args.epochs, max_ep_len=args.max_ep_len, 
                      learning_starts=args.learning_starts, train_freq=args.train_freq, 
                      target_network_update_freq=args.target_network_update_freq, 
                      gamma=args.gamma, q_lr=args.q_lr, logger_kwargs=logger_kwargs, 
                      save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)