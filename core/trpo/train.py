import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers.vector import RescaleAction
from gymnasium.wrappers.utils import RunningMeanStd
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from core.trpo.models.mlp import MLPActorCritic
from core.trpo.models.cnn import CNNActorCritic
from core.rl_utils import SkipAndScaleObservation, save_env
from core.utils import serialize_locals, clear_logs

np_eps = 1e-8 
class TRPOBuffer:
    def __init__(self, env: AsyncVectorEnv, buf_size, gamma, lam):
        # Check the type of the action space
        if not (isinstance(env.single_action_space, Discrete) or 
                isinstance(env.single_action_space, Box)):
            raise NotImplementedError
        
        # Initialize and store needed parameters 
        self.gamma, self.lam = gamma, lam
        self.buf_size = buf_size
        self.ep_start = np.zeros(env.num_envs, dtype=np.int64)
        self.obs_shape = env.single_observation_space.shape
        self.act_shape = env.single_action_space.shape
        env_buf_size = buf_size // env.num_envs
        self.rtg_rms = RunningMeanStd(dtype=np.float32)

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
        
        # Normalize returns to go
        self.rtg_rms.update(self.rtg.reshape(-1))
        rtg_mean, rtg_std = self.rtg_rms.mean, np.sqrt(self.rtg_rms.var)
        self.rtg = (self.rtg - rtg_mean)/(rtg_std + 1e-8)
        
        # Normalize advantages
        adv_mean, adv_std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - adv_mean)/adv_std
    
    def get_tensors(self, device):
        # Return reshaped tensors from buffer in the correct shape (combine envs experiences)
        to_tensor = lambda np_arr: torch.as_tensor(np_arr, dtype=torch.float32, device=device)

        return to_tensor(self.obs.reshape((-1,)+self.obs_shape)), \
                to_tensor(self.act.reshape((-1,)+self.act_shape)), \
                to_tensor(self.adv.reshape(-1)), \
                to_tensor(self.logp.reshape(-1)), \
                to_tensor(self.rtg.reshape(-1))

class ConjugateGradient:
    def __init__(self, damping_coeff, cg_iters):
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.kl_grad_flat = None        

    def conj_grad(self, policy_grad: np.ndarray, actor, device):
        # Calculate the gradient of KL-Divergence
        kl = actor.kl_divergence_grad()
        kl_grad = autograd.grad(outputs=kl, inputs=actor.parameters(), create_graph=True, retain_graph=True)
        kl_grad = torch.cat([g.flatten() for g in kl_grad])

        # Initialize parameters for CG algorithm
        x = np.zeros_like(policy_grad)
        r = policy_grad.copy()
        p = r.copy()
        r_dot_prev = np.dot(r, r)

        # Perform cg_iters of the CG algorithm to estimate x = inv(H) * policy_grad
        for _ in range(self.cg_iters):
            Hp = self.mat_vec_prod(kl_grad, torch.as_tensor(p, dtype=torch.float32, device=device), 
                                   actor, retain_graph=True)
            alpha = r_dot_prev / (np.dot(p, Hp) + np_eps)
            x += alpha * p
            r -= alpha * Hp
            r_dot_curr = np.dot(r, r)
            p = r + (r_dot_curr / r_dot_prev) * p
            r_dot_prev = r_dot_curr

        # Return final estimate of x, Hx and the KL-Divergence
        Hx = self.mat_vec_prod(kl_grad, torch.as_tensor(x, dtype=torch.float32, device=device), 
                               actor, retain_graph=False)
        
        return x, Hx 

    def mat_vec_prod(self, kl_grad: torch.Tensor, v: torch.Tensor,
                      actor, retain_graph=False):
        out = torch.dot(kl_grad, v)
        grad = autograd.grad(outputs=out, inputs=actor.parameters(), 
                             retain_graph=retain_graph)
        grad = torch.cat([g.flatten() for g in grad]) + self.damping_coeff * v

        return grad.detach().cpu().numpy()


class PolicyOptimizer:
    def __init__(self, delta, surr_obj_min, backtrack_iters, backtrack_coeff):
        self.delta = delta
        self.surr_obj_min = surr_obj_min
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_fail_ctr = 0

    def update_policy(self, epoch, obs, act, logp, adv, x: np.ndarray, 
                      Hx: np.ndarray, actor, writer: SummaryWriter):
        # Store parameters of current policy and policy itself for kl divergence
        pi_curr = actor.copy_policy()
        params_curr = nn.utils.parameters_to_vector(actor.parameters())

        # Perform the backtracking line search
        max_step = np.sqrt((2 * self.delta)/(np.dot(x, Hx) + np_eps))
        for i in range(self.backtrack_iters):
            offset = self.backtrack_coeff**i * max_step * x
            nn.utils.vector_to_parameters(params_curr + torch.as_tensor(
                offset, dtype=torch.float32, device=params_curr.device), actor.parameters())
            surr_obj = actor.surrogate_obj(obs, act, adv, logp)
            kl = actor.kl_divergence_no_grad(pi_curr)

            if (surr_obj > self.surr_obj_min) and (kl <= self.delta):
                writer.add_scalar('Pi/BacktrackIters', i, epoch+1)
                writer.add_scalar('Pi/KL', kl.item(), epoch+1)
            
                return True, surr_obj
                    
        # Backtracking failed so reload old params
        nn.utils.vector_to_parameters(params_curr, actor.parameters())
        writer.add_scalar('Pi/BacktrackIters', self.backtrack_iters, epoch+1)
        writer.add_scalar('Pi/KL', 0, epoch+1)
        self.backtrack_fail_ctr += 1
        
        return False, np.zeros(1)
            

class TRPOTrainer:
    def __init__(self, env_fn, wrappers_kwargs=dict(), use_gpu=False, model_path='', 
                 ac=MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=1000, 
                 gamma=0.99, delta=0.01, surr_obj_min=0.0, lr=1e-3, lr_f=None, 
                 max_grad_norm=0.5, clip_grad=True, train_iters=80, damping_coeff=0.1, 
                 cg_iters=10, backtrack_iters=10, backtrack_coeff=0.8, lam=0.95, 
                 log_dir=None, save_freq=10, checkpoint_freq=25):
        # Store needed hyperparameters
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.lr_f = lr_f
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad
        self.train_iters = train_iters
        self.save_freq = save_freq
        self.checkpoint_freq = checkpoint_freq

        # Serialize local hyperparameters
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn'); locals_dict.pop('wrappers_kwargs')
        locals_dict = serialize_locals(locals_dict)

        # Initialize logger and save hyperparameters
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(self.writer.get_logdir())}')
        self.save_dir = os.path.join(self.writer.get_logdir(), 'pyt_save')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize environment and attempt to save a copy of it 
        self.env = AsyncVectorEnv(env_fn)
        self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0) \
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
        self.writer.add_graph(self.ac_mod, torch.randn(size=self.env.observation_space.shape))

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
        self.ac_mod.to(self.device)

        # Initialize rest of objects needed for training
        self.buf = TRPOBuffer(self.env, steps_per_epoch * self.env.num_envs, gamma, lam)
        self.cg_mod = ConjugateGradient(damping_coeff, cg_iters)

        # Initialize optimizers
        self.pi_optim = PolicyOptimizer(delta, surr_obj_min, backtrack_iters, backtrack_coeff)
        self.val_optim = Adam(self.ac_mod.critic.parameters(), lr=lr)
    
    def __calc_policy_grad(self, obs, act, logp, adv):
        # Calculate policy loss
        loss_pi = self.__calc_policy_loss(obs, act, logp, adv)

        policy_grad = autograd.grad(loss_pi, self.ac_mod.actor.parameters(), 
                                    create_graph=True, retain_graph=True)
        policy_grad = torch.cat([g.flatten() for g in policy_grad]).detach().cpu().numpy()

        return policy_grad
    
    def __calc_policy_loss(self, obs, act, logp, adv):
        logp_grad = self.ac_mod.actor.log_prob_grad(obs, act)
        ratio = torch.exp(logp_grad - logp)
        policy_loss = (ratio * adv).mean()

        return policy_loss
    
    def __calc_val_loss(self, obs, rtg):
        val = self.ac_mod.critic.forward_grad(obs)
        val_loss = F.mse_loss(val, rtg)

        return val_loss
    
    def __update_params(self, epoch):
        # Get all collected experiences from buffer as tensors
        obs, act, adv, logp, rtg = self.buf.get_tensors(self.device)

        for _ in range(self.train_iters):
            # Calculate policy gradient and update policy
            policy_grad = self.__calc_policy_grad(obs, act, logp, adv)
            x, Hx = self.cg_mod.conj_grad(policy_grad, self.ac_mod.actor, self.device)
            success, loss_pi = self.pi_optim.update_policy(epoch, obs, act, 
                                                                     logp, adv, x, Hx, 
                                                                     self.ac_mod.actor, 
                                                                     self.writer)

            # Update value function
            self.val_optim.zero_grad()
            loss_val = self.__calc_val_loss(obs, rtg)
            loss_val.backward()
            if self.clip_grad == True:
                torch.nn.utils.clip_grad_norm_(self.ac_mod.critic.parameters(), self.max_grad_norm)
            self.val_optim.step()

            # Stop updates early to prevent large deviations from old policy 
            if success == False:
                break

        # Log epoch statistics
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        self.writer.add_scalar('Loss/LossV', loss_val.item(), epoch+1)
    
    def train_mod(self, epochs=100):
        # Initialize scheduler
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        val_scheduler = LinearLR(self.val_optim, start_factor=1.0, end_factor=end_factor, 
                                 total_iters=epochs)

        # Initialize environment variables
        obs, _ = self.env.reset(seed=self.seed)
        ep_len, ep_ret = np.zeros(self.env.num_envs, dtype=np.int64), 0
        ep_lens, ep_rets = [], []
        start_time = time.time()
        autoreset = np.zeros(self.env.num_envs)

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                act, val, logp = self.ac_mod.step(torch.as_tensor(obs, dtype=torch.float32, 
                                                                  device=self.device))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                               val[env_id], logp[env_id], step)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (self.steps_per_epoch-1)
                autoreset = np.logical_or(terminated, truncated)

                if np.any(autoreset):
                    for env_id in range(self.env.num_envs):
                        if autoreset[env_id]:
                            val_terminal = 0 if terminated[env_id] else self.ac_mod.critic(torch.as_tensor(
                                obs[env_id][None], dtype=torch.float32, device=self.device)).cpu().numpy()
                            ep_ret = self.buf.terminate_ep(env_id, ep_len[env_id], val_terminal)
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret)
                            ep_len[env_id] = 0
                
                if epoch_done:
                    obs, _ = self.env.reset()
                    self.buf.terminate_epoch()
                    ep_len = np.zeros_like(ep_len)
            
            self.__update_params(epoch)
            val_scheduler.step()
            
            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))
    
            
            # Log info about epoch
            if len(ep_rets) > 0:
                total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
                ep_lens, ep_rets = np.array(ep_lens), np.array(ep_rets)
                self.writer.add_scalar('EpLen/mean', ep_lens.mean(), total_steps_so_far)
                self.writer.add_scalar('EpRet/mean', ep_rets.mean(), total_steps_so_far)
                self.writer.add_scalar('EpRet/max', ep_rets.max(), total_steps_so_far)
                self.writer.add_scalar('EpRet/min', ep_rets.min(), total_steps_so_far)
                ep_lens, ep_rets = [], []
            self.writer.add_scalar('VVals/mean', self.buf.val.mean(), epoch+1)
            self.writer.add_scalar('VVals/max', self.buf.val.max(), epoch+1)
            self.writer.add_scalar('VVals/min', self.buf.val.min(), epoch+1)
            self.writer.add_scalar('Time', time.time()-start_time, epoch+1)
            self.writer.flush()
        
        # Save final model
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        self.env.close()
        print(f'Backtracking line search failed {self.pi_optim.backtrack_fail_ctr} times in total')
        print(f'Model {epochs} (final) saved successfully')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model and environment configuration
    parser.add_argument('--policy', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
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
    parser.add_argument('--log_std_init', nargs='+', type=float, default=[0]) # Used for MLP too

    # Rest of training arguments
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--surr_obj_min', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_f', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=bool, default=True)
    parser.add_argument('--train_iters', type=int, default=10)
    parser.add_argument('--damping_coeff', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--backtrack_iters', type=int, default=10)
    parser.add_argument('--backtrack_coeff', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='trpo')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Remove existing logs if run already exists
    clear_logs(log_dir)
    
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
    elif args.policy == 'cnn':
        ac = CNNActorCritic
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides, 
                         features_out=args.features_out,
                         log_std_init=args.log_std_init)
        env_fn_def = lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                       render_mode=render_mode)
        env_fn = [lambda render_mode=None: FrameStackObservation(SkipAndScaleObservation(
            GrayscaleObservation(env_fn_def(render_mode=render_mode))), stack_size=args.in_channels)] * args.cpu
        wrappers_kwargs = {
            'FrameStackObservation': {'stack_size': args.in_channels}
        }
    else:
        raise NotImplementedError

    # Begin training
    trainer = TRPOTrainer(env_fn, wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                          model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, seed=args.seed, 
                          steps_per_epoch=args.steps, gamma=args.gamma, delta=args.delta, 
                          surr_obj_min=args.surr_obj_min, lr=args.lr, lr_f=args.lr_f, 
                          max_grad_norm=args.max_grad_norm, clip_grad=args.clip_grad, 
                          train_iters=args.train_iters, damping_coeff=args.damping_coeff, 
                          cg_iters=args.cg_iters, backtrack_iters=args.backtrack_iters, 
                          backtrack_coeff=args.backtrack_coeff, lam=args.lam, 
                          log_dir=log_dir, save_freq=args.save_freq, 
                          checkpoint_freq=args.checkpoint_freq)
    
    trainer.train_mod(args.epochs)