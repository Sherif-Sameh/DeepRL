import os
import glob
import inspect
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
import time
import numpy as np
import torch
from torch import nn
from torch import autograd
import torch.distributions
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
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

np_eps = 1e-8 # np.finfo(np.float32).eps

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
        adv_mean, adv_std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - adv_mean)/adv_std
        self.ep_start = np.zeros_like(self.ep_start)


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

    def update_policy(self, epoch, x: np.ndarray, Hx: np.ndarray, actor, 
                      buf: TRPOBuffer, writer: SummaryWriter, device):
        # Store parameters of current policy and policy itself for kl divergence
        if isinstance(actor.pi, torch.distributions.Categorical):
            pi_curr = torch.distributions.Categorical(logits=actor.pi.logits)
        elif isinstance(actor.pi, torch.distributions.Normal):
            pi_curr = torch.distributions.Normal(loc=actor.pi.mean, scale=actor.pi.stddev)
        params_curr = nn.utils.parameters_to_vector(actor.parameters())

        # Perform the backtracking line search
        max_step = np.sqrt((2 * self.delta)/(np.dot(x, Hx) + np_eps))
        for i in range(self.backtrack_iters):
            offset = self.backtrack_coeff**i * max_step * x
            nn.utils.vector_to_parameters(params_curr + torch.as_tensor(offset, dtype=torch.float32, device=device), 
                                          actor.parameters())
            surr_obj = actor.surrogate_obj(torch.as_tensor(buf.obs.reshape((-1,)+buf.obs_shape), 
                                                           dtype=torch.float32, device=device),
                                           torch.as_tensor(buf.act.reshape((-1,)+buf.act_shape), 
                                                           dtype=torch.float32, device=device),
                                           buf.adv.reshape(-1), buf.logp.reshape(-1))
            
            kl = actor.kl_divergence_no_grad(pi_curr)
            if (surr_obj > self.surr_obj_min) and (kl <= self.delta):
                writer.add_scalar('Pi/BacktrackIters', i, epoch+1)
                writer.add_scalar('Pi/KL', kl.item(), epoch+1)
            
                return surr_obj
                    
        # Backtracking failed so reload old params
        nn.utils.vector_to_parameters(params_curr, actor.parameters())
        writer.add_scalar('Pi/BacktrackIters', self.backtrack_iters, epoch+1)
        writer.add_scalar('Pi/KL', 0, epoch+1)
        self.backtrack_fail_ctr += 1
        
        return np.zeros(1)
            

class TRPOTrainer:
    def __calc_policy_grad(self, actor, buf: TRPOBuffer, device):
        log_prob = actor.log_prob_grad(torch.as_tensor(buf.obs.reshape((-1,)+buf.obs_shape), 
                                                       dtype=torch.float32, device=device),
                                       torch.as_tensor(buf.act.reshape((-1,)+buf.act_shape), 
                                                       dtype=torch.float32, device=device))
        ratio = torch.exp(log_prob - torch.as_tensor(buf.logp.reshape(-1), dtype=torch.float32,
                                                     device=device))
        policy_loss = (ratio * torch.as_tensor(buf.adv.reshape(-1), dtype=torch.float32, 
                                               device=device)).mean()
        policy_grad = autograd.grad(policy_loss, actor.parameters(), create_graph=True, retain_graph=True)
        policy_grad = torch.cat([g.flatten() for g in policy_grad]).detach().cpu().numpy()

        return policy_grad
    
    def __calc_val_loss(self, critic, buf: TRPOBuffer, device):
        val = critic.forward_grad(torch.as_tensor(buf.obs.reshape((-1,)+buf.obs_shape), 
                                                  dtype=torch.float32, device=device))

        return ((torch.as_tensor(buf.rtg.reshape(-1), dtype=torch.float32, device=device) - val)**2).mean()
    
    def __update_params(self, device, epoch, train_v_iters, ac_mod: MLPActorCritic, 
                        buf: TRPOBuffer, cg_mod: ConjugateGradient, 
                        pi_optim: PolicyOptimizer,  val_optim: Adam, 
                        writer: SummaryWriter):
        # Update policy
        policy_grad = self.__calc_policy_grad(ac_mod.actor, buf, device)
        x, Hx = cg_mod.conj_grad(policy_grad, ac_mod.actor, device)
        loss_pi = pi_optim.update_policy(epoch, x, Hx, ac_mod.actor, buf, writer, device)

        # Update value function
        for i in range(train_v_iters):
            val_optim.zero_grad()
            loss_val = self.__calc_val_loss(ac_mod.critic, buf, device)
            loss_val.backward()
            val_optim.step()

        # Log epoch statistics
        writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        writer.add_scalar('Loss/LossV', loss_val.item(), epoch+1)
    
    def train_mod(self, env_fn, use_gpu=False, model_path='', ac=MLPActorCritic, 
                  ac_kwargs=dict(), seed=0, steps_per_epoch=4000, epochs=50, 
                  gamma=0.99, delta=0.01, surr_obj_min=0.0, lr=1e-3, lr_f=None, 
                  train_v_iters=80, damping_coeff=0.1, cg_iters=10, 
                  backtrack_iters=10, backtrack_coeff=0.8, lam=0.97, 
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
        writer.add_graph(ac_mod, torch.randn(size=env.observation_space.shape))

        local_steps_per_epoch = steps_per_epoch // env.num_envs

        # Setup random seed number for PyTorch and NumPy
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        # GPU setup if necessary
        if use_gpu == True:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.manual_seed(seed=seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
        ac_mod.to(device)

        # Initialize rest of objects needed for training
        buf_mod = TRPOBuffer(env, steps_per_epoch, gamma, lam)
        cg_mod = ConjugateGradient(damping_coeff, cg_iters)

        # Initialize optimizers and schedulers
        pi_optim = PolicyOptimizer(delta, surr_obj_min, backtrack_iters, backtrack_coeff)
        val_optim = Adam(ac_mod.critic.parameters(), lr=lr)
        end_factor = lr_f/lr if lr_f is not None else 1.0
        val_scheduler = LinearLR(val_optim, start_factor=1.0, end_factor=end_factor, 
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
                                torch.as_tensor(obs[env_id][None], dtype=torch.float32, device=device)).cpu().numpy()
                            ep_ret = buf_mod.terminate_ep(env_id, ep_len[env_id], val_terminal)
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret)
                            ep_len[env_id] = 0
                
                if epoch_done:
                    obs, _ = env.reset()
                    buf_mod.terminate_epoch()
                    ep_len = np.zeros_like(ep_len)
            
            self.__update_params(device, epoch, train_v_iters, ac_mod, buf_mod, 
                                 cg_mod, pi_optim, val_optim, writer)
            val_scheduler.step()
            
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
        print(f'Backtracking line search failed {pi_optim.backtrack_fail_ctr} times in total')
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--surr_obj_min', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_f', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--damping_coeff', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--backtrack_iters', type=int, default=10)
    parser.add_argument('--backtrack_coeff', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='trpo')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()

    # Set directory for logging
    log_dir = os.getcwd() + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Remove existing logs if run already exists
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        print('Warning: run already exists. Deleting previous logs... \n')
        files = glob.glob(os.path.join(log_dir, 'events.*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f'Failed to delete {f}. Reason {e}')

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
    trainer = TRPOTrainer()
    trainer.train_mod(env_fn, use_gpu=args.use_gpu, model_path=args.model_path, ac=ac, 
                      ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                      epochs=args.epochs, gamma=args.gamma, delta=args.delta, 
                      surr_obj_min=args.surr_obj_min, lr=args.lr, lr_f=args.lr_f,
                      train_v_iters=args.train_v_iters, damping_coeff=args.damping_coeff, 
                      cg_iters=args.cg_iters, backtrack_iters=args.backtrack_iters, 
                      backtrack_coeff=args.backtrack_coeff, lam=args.lam, 
                      log_dir=log_dir, save_freq=args.save_freq, 
                      checkpoint_freq=args.checkpoint_freq)