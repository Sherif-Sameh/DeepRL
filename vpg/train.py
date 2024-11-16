import gym
from gym.spaces import Box, Discrete
import time
import numpy as np
import torch
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, mpi_fork, mpi_statistics_scalar, num_procs

from models import MLPActorCritic

class VPGBuffer:
    def __init__(self, env, buf_size, num_epochs, gamma=0.98, lam=0.92):
        if isinstance(env.action_space, Discrete):
            is_discrete = True
        elif isinstance(env.action_space, Box):
            is_discrete = False
        else:
            raise NotImplementedError
        
        self.gamma = gamma
        self.lam = lam
        self.ep_start, self.epoch_ctr = 0, 0
        obs_dim = env.observation_space.shape[0]
        act_dim = 1 if is_discrete else env.action_space.shape[0]
        
        self.obs = np.zeros((buf_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((buf_size, act_dim), dtype=np.float32)
        self.rew = np.zeros((buf_size+1,), dtype=np.float32)
        self.rtg = np.zeros((buf_size,), dtype=np.float32)
        self.adv = np.zeros((buf_size,), dtype=np.float32)
        self.val = np.zeros((buf_size,), dtype=np.float32)
        self.logp = np.zeros((buf_size,), dtype=np.float32) 
        self.epoch_ret = np.zeros((num_epochs), dtype=np.float32)        

    def update_buffer(self, obs, act, rew, val, logp, tr_ctr):
        self.obs[self.ep_start + tr_ctr] = obs
        self.act[self.ep_start + tr_ctr] = act
        self.rew[self.ep_start + tr_ctr] = rew
        self.val[self.ep_start + tr_ctr] = val
        self.logp[self.ep_start + tr_ctr] = logp

    def terminate_ep(self, ep_len, val_terminal=0, epoch_done=True):
        # Calculate per episode statistics - Return to Go 
        self.rtg[self.ep_start+ep_len-1] = self.rew[self.ep_start+ep_len-1] + self.gamma*val_terminal
        for i in range(ep_len-2, -1, -1):
            self.rtg[self.ep_start+i] = self.rew[self.ep_start+i] + self.gamma*self.rtg[self.ep_start+i+1]
                                               
        # Calculate per episode statistics - Advantage function
        ep_slice = slice(self.ep_start, self.ep_start+ep_len)
        rews = np.append(self.rew[ep_slice], val_terminal)
        vals = np.append(self.val[ep_slice], val_terminal)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[self.ep_start+ep_len-1] = deltas[-1]
        for i in range(ep_len-2, -1, -1):
            self.adv[self.ep_start+i] = deltas[i] + self.gamma * self.lam * self.adv[self.ep_start+i+1]
        # self.adv[ep_slice] = self.rtg[ep_slice] - self.val[ep_slice]
        
        ep_ret = np.sum(self.rew[ep_slice])
        self.ep_start += ep_len
        if epoch_done:
            adv_mean, adv_std = mpi_statistics_scalar(self.adv)
            self.adv = (self.adv - adv_mean)/adv_std
            self.epoch_ret[self.epoch_ctr] = np.sum(self.rew)
            self.epoch_ctr += 1
            self.ep_start = 0

        return ep_ret
    
class VPGTrainer:
    def __calc_pi_loss(self, ac_mod: MLPActorCritic, buf: VPGBuffer):
        logp = ac_mod.actor.log_prob_grad(torch.as_tensor(buf.obs, dtype=torch.float32),
                                          torch.as_tensor(buf.act, dtype=torch.float32))
        loss_pi = -(logp * torch.as_tensor(buf.adv, dtype=torch.float32)).mean()

        # Useful extra info
        ent_pi = ac_mod.actor.pi.entropy().mean().item()

        return loss_pi, ent_pi

    def __calc_val_loss(self, ac_mod: MLPActorCritic, buf: VPGBuffer):
        val = ac_mod.critic.forward_grad(torch.as_tensor(buf.obs, dtype=torch.float32)) 
        
        return ((val - torch.as_tensor(buf.rtg, dtype=torch.float32))**2).mean()
        
    def __update_params(self, train_v_iters, ac_mod: MLPActorCritic, 
                        pi_optim: Adam, val_optim: Adam, 
                        logger: EpochLogger, buf: VPGBuffer):
        # Peform policy update
        pi_optim.zero_grad()
        loss_pi_old, ent_pi = self.__calc_pi_loss(ac_mod, buf)
        loss_pi_old.backward()
        mpi_avg_grads(ac_mod.actor)
        pi_optim.step()
        loss_pi, _ = self.__calc_pi_loss(ac_mod, buf)

        # Perform value function updates
        loss_val_old = self.__calc_val_loss(ac_mod, buf)
        for i in range(train_v_iters):
            val_optim.zero_grad()
            loss_val = self.__calc_val_loss(ac_mod, buf)
            loss_val.backward()
            mpi_avg_grads(ac_mod.critic)
            val_optim.step()

        # Log epoch statistics
        logp = ac_mod.actor.log_prob_no_grad(torch.as_tensor(buf.act))
        approx_kl = np.mean(buf.logp - logp).item()
        logger.store(LossPi=loss_pi_old.item(), LossV=loss_val_old.item(),
                     KL=approx_kl, Entropy=ent_pi,
                     DeltaLossPi=(loss_pi - loss_pi_old).item(),
                     DeltaLossV=(loss_val - loss_val_old).item())


    def train_mod(self, env_fn, ac=MLPActorCritic, ac_kwargs=dict(), seed=100, 
                  buf_size=4000, max_ep_len=1000, num_epochs=50, gamma=0.99, 
                  lam=0.97, pi_lr=3e-4, val_lr=1e-3, train_v_iters=80, 
                  save_freq=10, logger_kwargs=dict()):
        setup_pytorch_for_mpi()
        if num_procs() > 1:
            seed = seed * proc_id()    
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = env_fn()
        ac_mod = ac(env, **ac_kwargs)
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        local_buf_size = buf_size//num_procs()
        buf = VPGBuffer(env, local_buf_size, num_epochs, 
                        gamma=gamma, lam=lam)
        
        sync_params(ac_mod)
        pi_optim = Adam(ac_mod.actor.parameters(), lr=pi_lr)
        val_optim = Adam(ac_mod.critic.parameters(), lr=val_lr)
        logger.setup_pytorch_saver(ac_mod)

        obs = env.reset()
        ep_len, ep_ret, log_itr = 0, 0, 0
        start_time = time.time()

        for epoch in range(num_epochs):
            for step in range(local_buf_size):
                act, val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                
                obs_next, rew, done, _ = env.step(act)
                buf.update_buffer(obs, act, rew, val, logp, ep_len)
                logger.store(VVals=val)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_buf_size-1)
                max_ep_len_reached = ep_len == max_ep_len

                if epoch_done or max_ep_len_reached or done:
                    val_terminal = 0 if done else ac_mod.critic(torch.as_tensor(obs, dtype=torch.float32))
                    ep_ret = buf.terminate_ep(ep_len, val_terminal, epoch_done)
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs = env.reset()
                    ep_len, ep_ret = 0, 0
            
            if (epoch % save_freq) == 0:
                logger.save_state({'env': env})
                log_itr += 1
                print(f'Model {epoch} saved successfully')
                
            self.__update_params(train_v_iters, ac_mod, pi_optim, val_optim, 
                                 logger, buf)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*buf_size)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        
        # Save final model
        logger.save_state({'env': env})
        print(f'Model {num_epochs} (final) saved successfully')
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid_act', type=int, default=64)
    parser.add_argument('--hid_cri', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--val_lr', type=float, default=1e-3)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--ep_max_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='vpg_custom')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, 
                                        data_dir='/home/sherif/user/python/DRL/data/vpg')
    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l, 
                    hidden_sizes_critic=[args.hid_cri]*args.l,
                    hidden_acts_actor=torch.nn.Tanh, 
                    hidden_acts_critic=torch.nn.Tanh)

    trainer = VPGTrainer()
    trainer.train_mod(lambda : gym.make(args.env), ac=MLPActorCritic, ac_kwargs=ac_kwargs, 
                    seed=args.seed, buf_size=args.steps, max_ep_len=args.ep_max_len, 
                    num_epochs=args.epochs, gamma=args.gamma, lam=args.lam, pi_lr=args.pi_lr, 
                    val_lr=args.val_lr, save_freq=args.save_freq, logger_kwargs=logger_kwargs)