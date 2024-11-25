import os
import gym
from gym.spaces import Discrete, Box
import time
import numpy as np
import torch
from torch import nn
from torch import autograd
import torch.distributions
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, mpi_fork, mpi_statistics_scalar, num_procs, mpi_avg
from spinup.utils.run_utils import setup_logger_kwargs

from models import MLPActorCritic, MLPActor, MLPCritic

np_eps = 1e-8 # np.finfo(np.float32).eps

class TRPOBuffer:
    def __init__(self, env: gym.Env, buf_size, num_epochs, gamma, lam):
        # Check the type of the action space
        if not (isinstance(env.action_space, Discrete) or 
                isinstance(env.action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.gamma, self.lam = gamma, lam
        self.ep_start, self.epoch_ctr = 0, 0
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
        self.epoch_ret = np.zeros((num_epochs,), dtype=np.float32)
        self.surr_obj = np.zeros((num_epochs,), dtype=np.float32)

    def update_buffer(self, obs, act, rew, val, logp, tr_ctr):
        self.obs[self.ep_start + tr_ctr] = obs
        self.act[self.ep_start + tr_ctr] = act
        self.rew[self.ep_start + tr_ctr] = rew
        self.val[self.ep_start + tr_ctr] = val
        self.logp[self.ep_start + tr_ctr] = logp

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
        self.ep_start += ep_length

        # Finalize epoch statistics if epoch is done
        if epoch_done==True:
            adv_mean, adv_std = mpi_statistics_scalar(self.adv)
            self.adv = (self.adv - adv_mean)/adv_std
            self.epoch_ret[self.epoch_ctr] = np.sum(self.rew)
            self.ep_start = 0
            self.epoch_ctr += 1


class ConjugateGradient:
    def __init__(self, damping_coeff, cg_iters):
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.kl_grad_flat = None        

    def conj_grad(self, policy_grad: np.ndarray, log_prob: torch.Tensor,
                  actor: MLPActor, buf: TRPOBuffer):
        # Calculate the gradient of KL-Divergence
        kl = actor.kl_divergence_grad()
        kl_grad = autograd.grad(outputs=kl, inputs=actor.net.parameters(), create_graph=True, retain_graph=True)
        kl_grad = torch.cat([g.flatten() for g in kl_grad])

        # Initialize parameters for CG algorithm
        x = np.zeros_like(policy_grad)
        r = policy_grad.copy()
        p = r.copy()
        r_dot_prev = np.dot(r, r)

        # Perform cg_iters of the CG algorithm to estimate x = inv(H) * policy_grad
        for _ in range(self.cg_iters):
            Hp = self.mat_vec_prod(kl_grad, torch.as_tensor(p, dtype=torch.float32), actor, retain_graph=True)
            alpha = r_dot_prev / (np.dot(p, Hp) + np_eps)
            x += alpha * p
            r -= alpha * Hp
            r_dot_curr = np.dot(r, r)
            p = r + (r_dot_curr / r_dot_prev) * p
            r_dot_prev = r_dot_curr

        # Return final estimate of x, Hx and the KL-Divergence
        Hx = self.mat_vec_prod(kl_grad, torch.as_tensor(x, dtype=torch.float32), actor, retain_graph=False)
        
        return x, Hx 

    def mat_vec_prod(self, kl_grad: torch.Tensor, v: torch.Tensor,
                      actor: MLPActor, retain_graph=False):
        out = torch.dot(kl_grad, v)
        grad = autograd.grad(outputs=out, inputs=actor.net.parameters(), 
                             retain_graph=retain_graph)
        grad = torch.cat([g.flatten() for g in grad]) + self.damping_coeff * v

        return mpi_avg(grad.detach().numpy())


class PolicyOptimizer:
    def __init__(self, delta, surr_obj_min, backtrack_iters, backtrack_coeff):
        self.delta = delta
        self.surr_obj_min = surr_obj_min
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_fail_ctr = 0

    def update_policy(self, x: np.ndarray, Hx: np.ndarray, actor: MLPActor, 
                      buf: TRPOBuffer, logger: EpochLogger):
        # Store parameters of current policy and policy itself for kl divergence
        if isinstance(actor.pi, torch.distributions.Categorical):
            pi_curr = torch.distributions.Categorical(logits=actor.pi.logits)
        elif isinstance(actor.pi, torch.distributions.Normal):
            pi_curr = torch.distributions.Normal(loc=actor.pi.mean, scale=actor.pi.stddev)
        params_curr = nn.utils.parameters_to_vector(actor.net.parameters())

        # Perform the backtracking line search
        max_step = np.sqrt((2 * self.delta)/(np.dot(x, Hx) + np_eps))
        for i in range(self.backtrack_iters):
            offset = self.backtrack_coeff**i * max_step * x
            nn.utils.vector_to_parameters(params_curr + torch.as_tensor(offset, dtype=torch.float32), 
                                          actor.net.parameters())
            surr_obj = actor.surrogate_obj(torch.as_tensor(buf.obs, dtype=torch.float32),
                                           torch.as_tensor(buf.act, dtype=torch.float32),
                                           buf.adv, buf.logp)
            surr_obj = mpi_avg(surr_obj)
            
            kl = actor.kl_divergence_no_grad(pi_curr)
            kl = mpi_avg(kl)
            # print(f'i {i}: kl: {kl}')
            if (surr_obj > self.surr_obj_min) and (kl <= self.delta):
                logger.store(BacktrackIters=i, KL=kl.item())
                return surr_obj
                    
        # Backtracking failed so reload old params
        nn.utils.vector_to_parameters(params_curr, actor.net.parameters())
        logger.store(BacktrackIters=self.backtrack_iters, KL=0)
        self.backtrack_fail_ctr += 1
        
        return np.zeros(1)
            

class TRPOTrainer:
    def __calc_policy_grad(self, actor: MLPActor, buf: TRPOBuffer):
        log_prob = actor.log_prob_grad(torch.as_tensor(buf.obs, dtype=torch.float32),
                                       torch.as_tensor(buf.act, dtype=torch.float32))
        ratio = torch.exp(log_prob - torch.as_tensor(buf.logp, dtype=torch.float32))
        policy_loss = (ratio * torch.as_tensor(buf.adv, dtype=torch.float32)).mean()
        policy_grad = autograd.grad(policy_loss, actor.net.parameters(), create_graph=True, retain_graph=True)
        policy_grad = torch.cat([g.flatten() for g in policy_grad])
        policy_grad = mpi_avg(policy_grad.detach().numpy())

        return policy_grad, log_prob
    
    def __calc_val_loss(self, critic: MLPCritic, buf: TRPOBuffer):
        val = critic.forward_grad(torch.as_tensor(buf.obs, dtype=torch.float32))

        return ((torch.as_tensor(buf.rtg, dtype=torch.float32) - val)**2).mean()
    
    def __update_params(self, train_v_iters, ac_mod: MLPActorCritic, buf: TRPOBuffer, 
                        cg_mod: ConjugateGradient, pi_optim: PolicyOptimizer,  
                        val_optim: Adam, logger: EpochLogger):
        # Update policy
        policy_grad, log_prob = self.__calc_policy_grad(ac_mod.actor, buf)
        x, Hx = cg_mod.conj_grad(policy_grad, log_prob, ac_mod.actor, buf)
        loss_pi = pi_optim.update_policy(x, Hx, ac_mod.actor, buf, logger)
        buf.surr_obj[buf.epoch_ctr-1] = loss_pi
        loss_pi_old = 0 if buf.epoch_ctr==1 else buf.surr_obj[buf.epoch_ctr-2]

        # Update value function
        loss_val_old = self.__calc_val_loss(ac_mod.critic, buf).detach()
        for i in range(train_v_iters):
            val_optim.zero_grad()
            loss_val = self.__calc_val_loss(ac_mod.critic, buf)
            loss_val.backward()
            mpi_avg_grads(ac_mod.critic)
            val_optim.step()

        # Log epoch statistics
        logger.store(LossPi=loss_pi.item(), LossV=loss_val.item(),
                     DeltaLossPi=(loss_pi - loss_pi_old).item(),
                     DeltaLossV=(loss_val - loss_val_old).item())
    
    def train_mod(self, env_fn, model_path='', ac=MLPActorCritic, ac_kwargs=dict(), 
                  seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, 
                  surr_obj_min=0.003, vf_lr=1e-3, train_v_iters=80, 
                  damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
                  backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, 
                  logger_kwargs=dict(), save_freq=10, checkpoint_freq=20):
        setup_pytorch_for_mpi()
        
        # Initialize logger 
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals()) 
        
        local_steps_per_epoch = steps_per_epoch//num_procs()

        # Setup random seed number for PyTorch and NumPy
        seed += 10000 * proc_id()
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # Initialize environment and actor-critic
        env = env_fn()
        if len(model_path) > 0:
            ac_mod = torch.load(model_path)
        else:
            ac_mod = ac(env, **ac_kwargs)

        # Initialize rest of objects needed for training
        buf_mod = TRPOBuffer(env, local_steps_per_epoch, epochs, gamma, lam)
        cg_mod = ConjugateGradient(damping_coeff, cg_iters)

        # Sync AC parameters and initialize optimizers
        sync_params(ac_mod)
        pi_optim = PolicyOptimizer(delta, surr_obj_min, backtrack_iters, backtrack_coeff)
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
            if ((epoch + 1) % checkpoint_freq) == 0:
                logger.save_state({'env': env}, itr=epoch+1)
    
            self.__update_params(train_v_iters, ac_mod, buf_mod, cg_mod, 
                                 pi_optim, val_optim, logger)
            
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('BacktrackIters', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        
        # Save final model
        logger.save_state({'env': env})
        print(f'Backtracking line search failed {pi_optim.backtrack_fail_ctr} times in total')
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
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--surr_obj_min', type=float, default=0.0)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--damping_coeff', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--backtrack_iters', type=int, default=10)
    parser.add_argument('--backtrack_coeff', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='trpo_custom')
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
    data_dir = os.getcwd() + '/../data/trpo/' + args.env + '/'
    logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=data_dir, seed=args.seed)

    # Begin training
    trainer = TRPOTrainer()
    trainer.train_mod(lambda : gym.make(args.env), model_path=args.model_path, ac=MLPActorCritic, 
                      ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                      epochs=args.epochs, gamma=args.gamma, delta=args.delta, 
                      surr_obj_min=args.surr_obj_min, vf_lr=args.vf_lr, 
                      train_v_iters=args.train_v_iters, damping_coeff=args.damping_coeff, 
                      cg_iters=args.cg_iters, backtrack_iters=args.backtrack_iters, 
                      backtrack_coeff=args.backtrack_coeff, lam=args.lam, 
                      max_ep_len=args.max_ep_len, logger_kwargs=logger_kwargs, 
                      save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)