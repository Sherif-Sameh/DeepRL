import os
import inspect
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import Box, Discrete
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from models import MLPActorCritic

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

class VPGBuffer:
    def __init__(self, env: AsyncVectorEnv, buf_size, gamma=0.98, lam=0.92):
        if not (isinstance(env.single_action_space, Discrete) or 
                isinstance(env.single_action_space, Box)):
            raise NotImplementedError
        
        self.gamma, self.lam = gamma, lam
        self.buf_size = buf_size
        self.ep_start = np.zeros(env.num_envs, dtype=np.int64)
        obs_shape = env.single_observation_space.shape
        act_shape = env.single_action_space.shape
        env_buf_size = buf_size // env.num_envs

        self.obs = np.zeros((env.num_envs, env_buf_size) + obs_shape, dtype=np.float32)
        self.act = np.zeros((env.num_envs, env_buf_size) + act_shape, dtype=np.float32)
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


class VPGTrainer:
    def __calc_pi_loss(self, ac_mod: MLPActorCritic, buf: VPGBuffer):
        logp = ac_mod.actor.log_prob_grad(torch.as_tensor(buf.obs.reshape(buf.buf_size, -1), dtype=torch.float32),
                                          torch.as_tensor(buf.act.reshape(buf.buf_size, -1), dtype=torch.float32))
        loss_pi = -(logp * torch.as_tensor(buf.adv.reshape(-1), dtype=torch.float32)).mean()

        # Useful extra info
        ent_pi = ac_mod.actor.pi.entropy().mean().item()

        return loss_pi, ent_pi

    def __calc_val_loss(self, ac_mod: MLPActorCritic, buf: VPGBuffer):
        val = ac_mod.critic.forward_grad(torch.as_tensor(buf.obs.reshape(buf.buf_size, -1), dtype=torch.float32)) 
        
        return ((val - torch.as_tensor(buf.rtg.reshape(-1), dtype=torch.float32))**2).mean()
        
    def __update_params(self, train_v_iters, epoch, ac_mod: MLPActorCritic, pi_optim: Adam, 
                        val_optim: Adam, writer: SummaryWriter, buf: VPGBuffer):
        # Peform policy update
        pi_optim.zero_grad()
        loss_pi_old, ent_pi = self.__calc_pi_loss(ac_mod, buf)
        loss_pi_old.backward()
        pi_optim.step()

        # Perform value function updates
        loss_val_old = self.__calc_val_loss(ac_mod, buf)
        for i in range(train_v_iters):
            val_optim.zero_grad()
            loss_val = self.__calc_val_loss(ac_mod, buf)
            loss_val.backward()
            val_optim.step()

        # Log epoch statistics
        logp = ac_mod.actor.log_prob_no_grad(torch.as_tensor(buf.act.reshape(buf.buf_size, -1)))
        approx_kl = np.mean(buf.logp.reshape(-1) - logp.numpy()).item()
        writer.add_scalar('Loss/LossPi', loss_pi_old.item(), epoch)
        writer.add_scalar('Loss/LossV', loss_val_old.item(), epoch)
        writer.add_scalar('Pi/KL', approx_kl, epoch)
        writer.add_scalar('Pi/Entropy', ent_pi, epoch)


    def train_mod(self, env_fn, model_path='', ac=MLPActorCritic, ac_kwargs=dict(), 
                  seed=0, buf_size=4000, num_epochs=50, gamma=0.99, 
                  lam=0.97, pi_lr=3e-4, val_lr=1e-3, train_v_iters=80, 
                  log_dir=None, save_freq=10, checkpoint_freq=25):
        locals_dict = locals()
        locals_dict.pop('self'); locals_dict.pop('env_fn')
        locals_dict = serialize_locals(locals_dict)

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_hparams(locals_dict, {}, run_name=f'../{os.path.basename(writer.get_logdir())}')
        save_dir = os.path.join(writer.get_logdir(), 'pyt_save')
        os.makedirs(save_dir, exist_ok=True)
            
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = AsyncVectorEnv(env_fn)
        if len(model_path) > 0:
            ac_mod = torch.load(model_path)
        else:
            ac_mod = ac(env, **ac_kwargs)
        writer.add_graph(ac_mod, torch.randn(size=env.observation_space.shape))

        buf = VPGBuffer(env, buf_size, gamma=gamma, lam=lam)
        local_buf_size = buf_size // env.num_envs
        
        pi_optim = Adam(ac_mod.actor.parameters(), lr=pi_lr)
        val_optim = Adam(ac_mod.critic.parameters(), lr=val_lr)

        obs, _ = env.reset(seed=seed)
        ep_len, ep_ret = np.zeros(env.num_envs, dtype=np.int64), 0
        ep_lens, ep_rets = [], []
        start_time = time.time()
        autoreset = np.zeros(env.num_envs)

        for epoch in range(num_epochs):
            for step in range(local_buf_size):
                act, val, logp = ac_mod.step(torch.as_tensor(obs, dtype=torch.float32))
                obs_next, rew, terminated, truncated, _ = env.step(act)
                
                for env_id in range(env.num_envs):
                    if not autoreset[env_id]:
                        buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                          val[env_id], logp[env_id], step)
                obs, ep_len = obs_next, ep_len + 1

                epoch_done = step == (local_buf_size-1)
                autoreset = np.logical_or(terminated, truncated)

                if np.any(autoreset):
                    for env_id in range(env.num_envs):
                        if autoreset[env_id]:
                            val_terminal = 0 if terminated[env_id] else ac_mod.critic(
                                torch.as_tensor(obs[env_id], dtype=torch.float32)).numpy()
                            ep_ret = buf.terminate_ep(env_id, ep_len[env_id], val_terminal)
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret)
                            ep_len[env_id] = 0
                
                if epoch_done:
                    obs, _ = env.reset()
                    buf.terminate_epoch()
                    ep_len = np.zeros_like(ep_len)
            
            self.__update_params(train_v_iters, epoch+1, ac_mod, pi_optim, 
                                 val_optim, writer, buf)
            
            if (epoch % save_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
            if ((epoch + 1) % checkpoint_freq) == 0:
                torch.save(ac_mod, os.path.join(save_dir, f'model{epoch+1}.pt'))
                

            # Log info about epoch
            if len(ep_rets) > 0:
                ep_lens, ep_rets = np.array(ep_lens), np.array(ep_rets)
                writer.add_scalar('EpLen/mean', ep_lens.mean(), (epoch+1)*buf_size)
                writer.add_scalar('EpRet/mean', ep_rets.mean(), (epoch+1)*buf_size)
                writer.add_scalar('EpRet/max', ep_rets.max(), (epoch+1)*buf_size)
                writer.add_scalar('EpRet/min', ep_rets.min(), (epoch+1)*buf_size)
                ep_lens, ep_rets = [], []
            writer.add_scalar('VVals/mean', buf.val.mean(), epoch+1)
            writer.add_scalar('VVals/max', buf.val.max(), epoch+1)
            writer.add_scalar('VVals/min', buf.val.min(), epoch+1)
            writer.add_scalar('Time', time.time()-start_time, epoch+1)
            writer.flush()
        
        # Save final model
        torch.save(ac_mod, os.path.join(save_dir, 'model.pt'))
        writer.close()
        print(f'Model {num_epochs} (final) saved successfully')
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--model_path', type=str, default='')
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
    parser.add_argument('--max_ep_len', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    log_dir = os.getcwd() + '/../runs/vpg/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    ac_kwargs = dict(hidden_sizes_actor=[args.hid_act]*args.l, 
                    hidden_sizes_critic=[args.hid_cri]*args.l,
                    hidden_acts_actor=torch.nn.Tanh, 
                    hidden_acts_critic=torch.nn.Tanh)
    
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    env_fn = [lambda: gym.make(args.env, max_episode_steps=max_ep_len)] * args.cpu

    trainer = VPGTrainer()
    trainer.train_mod(env_fn, model_path=args.model_path, ac=MLPActorCritic, 
                      ac_kwargs=ac_kwargs, seed=args.seed, buf_size=args.steps, 
                      num_epochs=args.epochs, gamma=args.gamma, lam=args.lam, 
                      pi_lr=args.pi_lr, val_lr=args.val_lr, log_dir=log_dir, 
                      save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)