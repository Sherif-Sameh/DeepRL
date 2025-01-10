import os
import sys
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from gymnasium.wrappers import  AddRenderObservation, ResizeObservation
from gymnasium.wrappers.vector import NormalizeReward
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from core.ppo.train import PPOTrainer
from core.ppo.train import get_parser as get_parser_ppo
from core.td3.train import TD3Trainer
from core.td3.train import get_parser as get_parser_td3
from core.ppo.models.cnn_lstm import CNNLSTMActorCritic as PPOActorCritic
from core.td3.models.cnn_lstm import CNNLSTMActorCritic as TD3ActorCritic
from exploration.cdesp.models.icm import IntrinsicCuriosityModule
from core.rl_utils import SkipAndScaleObservation

class CDESPTrainer:
    def __init__(self, env, rew_icm_max, fwd_coeff, icm_kwargs=dict()):
        self.fwd_coeff = fwd_coeff
        self.rew_icm_max = rew_icm_max
        
        # Initialize ICM module
        self.icm_mod = IntrinsicCuriosityModule(env, **icm_kwargs)
        self.icm_mod.layer_summary()

    def _get_icm_loss_mask(self, loss_mask: torch.Tensor):
        # A valid transition for ICM is where both observations are sampled
        # from the same episode
        loss_mask_curr = loss_mask[:, :-1]
        loss_mask_next = loss_mask[:, 1:]
        loss_mask_icm = torch.logical_and(loss_mask_curr, loss_mask_next) 

        return loss_mask_icm

    def _calc_inv_loss_discrete(self, obs: torch.Tensor, act: torch.Tensor, 
                                loss_mask: torch.Tensor):
        act_preds, features = self.icm_mod.inv_mod.forward(obs) # action logits
        act_target = act[:, :-1].long()
        loss_mask_icm = self._get_icm_loss_mask(loss_mask)

        return F.cross_entropy(act_preds[loss_mask_icm], 
                               act_target[loss_mask_icm]), features
    
    def _calc_inv_loss_cont(self, obs: torch.Tensor, act: torch.Tensor, 
                            loss_mask: torch.Tensor):
        act_preds, features = self.icm_mod.inv_mod.forward(obs)
        act_target = act[:, :-1]
        loss_mask_icm = self._get_icm_loss_mask(loss_mask)
    
        return F.mse_loss(act_preds[loss_mask_icm], 
                          act_target[loss_mask_icm]), features
    
    def _calc_fwd_loss(self, features: torch.Tensor, act: torch.Tensor, 
                        loss_mask: torch.Tensor):
        features_pred = self.icm_mod.fwd_mod.forward(features, act)[:, :-1]
        features_target = features[:, 1:].detach()
        loss_mask_icm = self._get_icm_loss_mask(loss_mask)

        return F.mse_loss(features_pred[loss_mask_icm], 
                          features_target[loss_mask_icm])

class CDESP_PPOTrainer(CDESPTrainer, PPOTrainer):
    def __init__(self, rew_icm_max=1.0, fwd_coeff=0.2, lr_icm=1e-3, 
                 icm_kwargs=dict(), **ppo_args):
        # Initialize actor-critic trainer
        PPOTrainer.__init__(self, **ppo_args)
        
        # Initialize CDESP trainer and ICM module
        CDESPTrainer.__init__(self, self.env, rew_icm_max, fwd_coeff, icm_kwargs)
        self.icm_mod.to(self.device)

        # Re-initialize common optimizer with added modules from ICM
        self.ac_optim = Adam([{'params': self.ac_mod.parameters(), 'lr': self.lr},
                              {'params': self.icm_mod.fwd_mod.parameters(), 'lr': lr_icm},
                              {'params': self.icm_mod.inv_mod.parameters(), 'lr': lr_icm/2}])

        # Determine inverse model loss function based on environment action space type
        if isinstance(self.env.single_action_space, Discrete):
            self.inv_loss = self._calc_inv_loss_discrete
        elif isinstance(self.env.single_action_space, Box):
            self.inv_loss = self._calc_inv_loss_cont
    
    def _update_params(self, epoch):
        # Store old policy and all observations for KL early stopping
        obs_all = self.buf.get_all_obs(self.device) # Returns a tuple (obs, mask) for LSTM policies
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
        with torch.no_grad(): self.ac_mod.actor.forward(obs_all[0])
        pi_curr = self.ac_mod.actor.copy_policy()

        # Loop train_iters times over the whole dataset (unless early stopping occurs)
        kls = []
        for i in range(self.train_iters):
            # Get dataloader for performing mini-batch SGD updates
            dataloader = self.buf.get_dataloader(self.device)
            
            # Loop over dataset in mini-batches
            for obs, act, adv, logp, rtg, mask in dataloader:
                self.ac_optim.zero_grad()
                
                # Normalize advantages mini-batch wise
                adv_mean, adv_std = adv.mean(), adv.std()
                adv = (adv - adv_mean) / adv_std

                # Get policy, entropy and value losses
                self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
                loss_pi = self._calc_policy_loss(obs, act, adv, logp, loss_mask=mask)
                loss_ent = self._calc_entropy_loss(loss_mask=mask)
                self.ac_mod.reset_hidden_states(self.device, batch_size=self.buf.batch_size)
                loss_val = self._calc_val_loss(obs, rtg, loss_mask=mask)

                # Get inverse and forward model losses
                loss_inv, features = self.inv_loss(obs, act, mask)
                loss_fwd = self._calc_fwd_loss(features, act, mask)

                # Combine losses and compute gradients
                loss_ac = loss_pi + self.ent_coeff * loss_ent + self.vf_coeff * loss_val
                loss_icm = loss_inv + self.fwd_coeff * loss_fwd
                loss_ac.backward()
                loss_icm.backward()

                # Clip gradients (if required) and update parameters
                if self.clip_grad == True:
                    torch.nn.utils.clip_grad_norm_(self.ac_mod.parameters(), self.max_grad_norm)
                self.ac_optim.step()

            # Check KL-Divergence constraint for triggering early stopping
            self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
            with torch.no_grad(): kl = self.ac_mod.actor.kl_divergence(obs_all, pi_curr)
            kls.append(kl.item())
            if (self.target_kl is not None) and (kl > 1.5 * self.target_kl):
                # print(f'Actor updates cut-off after {i+1} iterations by KL {kl}')
                break

        # Log epoch statistics
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        self.writer.add_scalar('Loss/LossEnt', loss_ent.item(), epoch+1)
        self.writer.add_scalar('Loss/LossV', loss_val.item(), epoch+1)
        self.writer.add_scalar('Loss/LossInv', loss_inv.item(), epoch+1)
        self.writer.add_scalar('Loss/LossFwd', loss_fwd.item(), epoch+1)
        self.writer.add_scalar('Pi/KL', np.mean(kls), epoch+1)

    def train_mod(self, epochs=100):
        # Initialize scheduler
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        ac_scheduler = LinearLR(self.ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Normalize returns for more stable training
        if not isinstance(self.env, NormalizeReward):
            self.env = NormalizeReward(self.env, gamma=self.gamma)
        self._run_env(num_episodes=10)

        # Initialize environment variables
        obs, _ = self.env.reset(seed=self.seed)
        ep_len, ep_ret = np.zeros(self.env.num_envs, dtype=np.int64), np.zeros(self.env.num_envs)
        ep_lens, ep_rets = [], []
        autoreset = np.zeros(self.env.num_envs)
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)

        # Initialize ICM-specific variables
        ep_ret_icm = np.zeros_like(ep_ret)
        clipped_rew_icm = 0
        steps_per_update = self.eval_every * self.steps_per_epoch * self.env.num_envs

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                act, val, logp = self.ac_mod.step(to_tensor(obs))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)
                rew_icm = self.icm_mod.calc_reward(to_tensor(obs), to_tensor(obs_next),
                                                   to_tensor(act))
                rew_to_clip = rew_icm > self.rew_icm_max
                rew_icm[rew_to_clip] = self.rew_icm_max
                autoreset_next = np.logical_or(terminated, truncated)
                ep_len += 1

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]: 
                        rew[env_id] += rew_icm[env_id]
                        ep_ret_icm[env_id] += rew_icm[env_id]
                        clipped_rew_icm += rew_to_clip[env_id]
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], rew[env_id], 
                                               val[env_id], logp[env_id], step)
                    if autoreset_next[env_id]:
                        val_terminal = 0 if terminated[env_id] else self.ac_mod.get_terminal_value(
                            to_tensor(obs_next), env_id)
                        ep_ret[env_id] += self.buf.terminate_ep(env_id, min(ep_len[env_id], step+1), val_terminal)
                        ep_ret[env_id] *= np.sqrt(self.env.return_rms.var + self.env.epsilon)
                        ep_lens.append(ep_len[env_id])
                        # ep_rets.append(ep_ret[env_id] - ep_ret_icm[env_id])
                        ep_rets.append(ep_ret[env_id])
                        ep_len[env_id], ep_ret[env_id], ep_ret_icm[env_id] = 0, 0, 0
                        self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)
                obs, autoreset = obs_next, autoreset_next

            self.ac_mod.reset_hidden_states(self.device, save=True)
            for env_id in range(self.env.num_envs): 
                if ep_len[env_id] > 0: 
                    val_terminal = self.ac_mod.get_terminal_value(to_tensor(obs), env_id)
                    ep_ret[env_id] += self.buf.terminate_ep(env_id, min(ep_len[env_id], self.steps_per_epoch), 
                                                            val_terminal)
            self.buf.terminate_epoch()
            self._update_params(epoch)
            ac_scheduler.step()
            self.ac_mod.reset_hidden_states(self.device, restore=True)
            
            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))
                
            # Log info about epoch
            if ((epoch + 1) % self.eval_every) == 0:
                if len(ep_rets) > 0:
                    total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
                    ep_lens, ep_rets = np.array(ep_lens), np.array(ep_rets)
                    ratio_clipped_rew_icm = clipped_rew_icm/steps_per_update
                    self.writer.add_scalar('EpLen/mean', ep_lens.mean(), total_steps_so_far)
                    self.writer.add_scalar('EpRet/mean', ep_rets.mean(), total_steps_so_far)
                    self.writer.add_scalar('EpRet/max', ep_rets.max(), total_steps_so_far)
                    self.writer.add_scalar('EpRet/min', ep_rets.min(), total_steps_so_far)
                    ep_lens, ep_rets = [], []
                self.writer.add_scalar('VVals/mean', self.buf.val.mean(), epoch+1)
                self.writer.add_scalar('VVals/max', self.buf.val.max(), epoch+1)
                self.writer.add_scalar('VVals/min', self.buf.val.min(), epoch+1)
                self.writer.add_scalar('ClippedRewRatio', ratio_clipped_rew_icm, epoch+1)
                self.writer.flush()
                clipped_rew_icm = 0
        
        # Save final model
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        self.env.close()
        print(f'Model {epochs} (final) saved successfully')

class CDESP_TD3Trainer(CDESPTrainer, TD3Trainer):
    def __init__(self, rew_icm_max=1.0, fwd_coeff=0.2, lr_icm=1e-3, 
                 icm_kwargs=dict(), **td3_args):
        # Initialize actor-critic trainer
        TD3Trainer.__init__(self, **td3_args)
        
        # Initialize CDESP trainer and ICM module
        CDESPTrainer.__init__(self, self.env, rew_icm_max, fwd_coeff, icm_kwargs)
        self.icm_mod.to(self.device)

        # Re-initialize optimizer with added modules from ICM
        self.ac_optim = Adam([{'params': self.ac_mod.parameters(), 'lr': self.lr},
                              {'params': self.icm_mod.parameters(), 'lr': lr_icm}])
        
        # Choose inverse model loss function (always continuous)
        self.inv_loss = self._calc_inv_loss_cont
    
    def _update_params(self, epoch, update_policy):
        # Get mini-batch from replay buffer
        obs, act, rew, obs_next, done, mask = self.buf.get_batch(self.device)
        
        # Get critics loss
        self.ac_optim.zero_grad()
        loss_q1, loss_q2 = self._calc_q_loss(obs, act, rew, obs_next, done, mask)
        
        # Get inverse and forward model losses 
        loss_inv, features = self.inv_loss(obs, act, mask)
        loss_fwd = self._calc_fwd_loss(features, act, mask)

        if update_policy == True:
            # Get actor loss (critic's weights are frozen temporarily)
            self.ac_mod.critic_1.set_grad_tracking(val=False)
            loss_pi = self._calc_pi_loss(obs, mask)
            self.ac_mod.critic_1.set_grad_tracking(val=True)

            # Combine losses and calculate gradients
            loss_ac = loss_pi + 0.5 * (loss_q1 + loss_q2)
            loss_icm = loss_inv + self.fwd_coeff * loss_fwd
            loss_ac.backward()
            loss_icm.backward()

            # Clip gradients (if neccessary and update parameters)
            if self.clip_grad == True:
                torch.nn.utils.clip_grad_norm_(self.ac_mod.actor.parameters(), self.max_grad_norm)
            self.ac_optim.step()

            # Update actor target network
            self.ac_mod.actor.update_target(self.polyak)
            
            # Log training statistics
            self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        else:
            loss_ac = 0.5 * (loss_q1 + loss_q2)
            loss_icm = loss_inv + self.fwd_coeff * loss_fwd
            loss_ac.backward()
            loss_icm.backward()
            self.ac_optim.step()
        
        # Update target critic networks
        self.ac_mod.critic_1.update_target(self.polyak)
        self.ac_mod.critic_2.update_target(self.polyak)

        # Log training statistics
        self.writer.add_scalar('Loss/LossQ1', loss_q1.item(), epoch+1)
        self.writer.add_scalar('Loss/LossQ2', loss_q2.item(), epoch+1)
        self.writer.add_scalar('Loss/LossInv', loss_inv.item(), epoch+1)
        self.writer.add_scalar('Loss/LossFwd', loss_fwd.item(), epoch+1)

    
    def train_mod(self, epochs=100):
        # Initialize scheduler
        end_factor = self.lr_f/self.lr if self.lr_f is not None else 1.0
        ac_scheduler = LinearLR(self.ac_optim, start_factor=1.0, end_factor=end_factor, 
                                total_iters=epochs)
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Normalize returns for more stable training
        if not isinstance(self.env, NormalizeReward):
            self.env = NormalizeReward(self.env, gamma=self.gamma)
            
        # Initialize environment variables
        obs, _ = self.env.reset(seed=self.seed)
        autoreset = np.zeros(self.env.num_envs)
        self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
        q_vals = []

        # Initialize ICM-specific variables
        clipped_rew_icm = 0
        steps_per_update = self.steps_per_epoch * self.env.num_envs

        for epoch in range(epochs):
            for step in range(self.steps_per_epoch):
                if (step + self.steps_per_epoch*epoch) > self.start_steps:
                    act, q_val = self.ac_mod.step(to_tensor(obs))
                else:
                    act = self.env.action_space.sample()
                    _, q_val = self.ac_mod.step(to_tensor(obs))
                obs_next, rew, terminated, truncated, _ = self.env.step(act)
                rew_icm = self.icm_mod.calc_reward(to_tensor(obs), to_tensor(obs_next),
                                                   to_tensor(act))
                rew_to_clip = rew_icm > self.rew_icm_max
                rew_icm[rew_to_clip] = self.rew_icm_max

                for env_id in range(self.env.num_envs):
                    if not autoreset[env_id]:
                        rew[env_id] += rew_icm[env_id]
                        clipped_rew_icm += rew_to_clip[env_id]
                        self.buf.update_buffer(env_id, obs[env_id], act[env_id], 
                                               rew[env_id], terminated[env_id])
                        q_vals.append(q_val[env_id])
                    else:
                        self.buf.increment_ep_num(env_id)
                        self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)
                obs = obs_next
                autoreset = np.logical_or(terminated, truncated)
                
                if (self.buf.get_buffer_size() >= self.learning_starts) \
                    and ((step % self.update_every) == 0):
                    self.ac_mod.reset_hidden_states(self.device, save=True)
                    for j in range(self.num_updates):
                        update_policy = (j % self.policy_delay) == 0 
                        self._update_params(epoch, update_policy)
                    self.ac_mod.reset_hidden_states(self.device, restore=True)
            
            # Evaluate deterministic policy (skip return normalization wrapper)
            env = self.env.env if isinstance(self.env, NormalizeReward) else self.env 
            ep_len, ep_ret = np.zeros(env.num_envs), np.zeros(env.num_envs)
            ep_lens, ep_rets = [], []
            obs, _ = env.reset()
            self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)
            while len(ep_lens) < self.num_test_episodes*env.num_envs:
                act = self.ac_mod.act(to_tensor(obs))
                obs, rew, terminated, truncated, _ = env.step(act)
                ep_len, ep_ret = ep_len + 1, ep_ret + rew
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    for env_id in range(env.num_envs):
                        if done[env_id]:
                            ep_lens.append(ep_len[env_id])
                            ep_rets.append(ep_ret[env_id])
                            ep_len[env_id], ep_ret[env_id] = 0, 0
                            self.ac_mod.reset_hidden_states(self.device, batch_idx=env_id)
            obs, _ = self.env.reset()
            ac_scheduler.step()
            self.ac_mod.step_action_std(epochs)
            self.ac_mod.reset_hidden_states(self.device, batch_size=self.env.num_envs)

            if ((epoch + 1) % self.save_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
            if ((epoch + 1) % self.checkpoint_freq) == 0:
                torch.save(self.ac_mod, os.path.join(self.save_dir, f'model{epoch+1}.pt'))

            # Log info about epoch
            total_steps_so_far = (epoch+1)*self.steps_per_epoch*self.env.num_envs
            ep_lens, ep_rets, q_vals = np.array(ep_lens), np.array(ep_rets), np.array(q_vals)
            ratio_clipped_rew_icm = clipped_rew_icm/steps_per_update
            self.writer.add_scalar('EpLen/mean', ep_lens.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/mean', ep_rets.mean(), total_steps_so_far)
            self.writer.add_scalar('EpRet/max', ep_rets.max(), total_steps_so_far)
            self.writer.add_scalar('EpRet/min', ep_rets.min(), total_steps_so_far)
            self.writer.add_scalar('QVals/mean', q_vals.mean(), epoch+1)
            self.writer.add_scalar('QVals/max', q_vals.max(), epoch+1)
            self.writer.add_scalar('QVals/min', q_vals.min(), epoch+1)
            self.writer.add_scalar('ClippedRewRatio', ratio_clipped_rew_icm, epoch+1)
            self.writer.flush()
            q_vals, clipped_rew_icm = [], 0
        
        # Save final model
        torch.save(self.ac_mod, os.path.join(self.save_dir, 'model.pt'))
        self.writer.close()
        self.env.close()
        print(f'Model {epochs} (final) saved successfully')

def get_algo_type():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'td3'])

    # Parse first argument and pass on remaining ones
    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    return args.algo, remaining_args

if __name__ == '__main__':
    # Parse first argument to know algorithm type
    algo, remaining_args = get_algo_type()

    # Get parser all remaining DRL algorithm and add CDESP arguments 
    if algo == 'ppo':
        parser = get_parser_ppo()
    elif algo == 'td3':
        parser = get_parser_td3()

    # ICM parameters
    parser.add_argument('--beta', type=float, default=0.02)
    parser.add_argument('--hid_inv', nargs='+', type=int, default=[256])
    parser.add_argument('--hid_fwd', nargs='+', type=int, default=[256])

    # Rest of training parameters
    parser.add_argument('--fwd_coeff', type=float, default=0.25)
    parser.add_argument('--rew_icm_max', type=float, default=1.0)
    parser.add_argument('--lr_icm', type=float, default=1e-3)

    args = parser.parse_args(remaining_args)
    args.exp_name = 'cdesp_'+ args.exp_name
    assert args.policy == 'cnn-lstm', "CDESP must be used with a CNN-LSTM policy"

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Setup actor-critic and ICM's kwargs
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if algo == 'ppo':
        ac = PPOActorCritic
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides,
                         padding=args.padding, 
                         features_out=args.features_out,
                         log_std_init=args.log_std_init)
    elif algo == 'td3':
        ac = TD3ActorCritic
        ac_kwargs = dict(in_channels=args.in_channels, 
                         out_channels=args.out_channels,
                         kernel_sizes=args.kernel_sizes, 
                         strides=args.strides, 
                         padding=args.padding,
                         features_out=args.features_out,
                         hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri,
                         action_std=args.action_std,
                         action_std_f=args.action_std_f)
    icm_kwargs = dict(beta=args.beta,
                      in_channels=args.in_channels, 
                      out_channels=args.out_channels,
                      kernel_sizes=args.kernel_sizes, 
                      strides=args.strides,
                      padding=args.padding, 
                      hidden_size_inv=args.hid_inv,
                      hidden_sizes_fwd=args.hid_fwd)
    
    # Setup environment lambda 
    # env_fn_def = lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
    #                                                    render_mode=render_mode)
    env_fn_def = lambda render_mode='rgb_array': ResizeObservation(AddRenderObservation(
        gym.make(args.env, max_episode_steps=max_ep_len, render_mode=render_mode)), shape=(42, 42))
    env_fn = [lambda render_mode='rgb_array': FrameStackObservation(SkipAndScaleObservation(
        GrayscaleObservation(env_fn_def(render_mode=render_mode)), skip=args.action_rep), 
        stack_size=args.in_channels)] * args.cpu
    wrappers_kwargs = {
        'ResizeObservation': {'shape': (42, 42)},
        'SkipAndScaleObservation': {'skip': args.action_rep},
        'FrameStackObservation': {'stack_size': args.in_channels}
    }

    # Setup the trainer and begin training
    if algo == 'ppo':
        trainer = CDESP_PPOTrainer(rew_icm_max=args.rew_icm_max, fwd_coeff=args.fwd_coeff, lr_icm=args.lr_icm,
                                   icm_kwargs=icm_kwargs, env_fn=env_fn, wrappers_kwargs=wrappers_kwargs, 
                                   use_gpu=args.use_gpu, model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
                                   seed=args.seed, steps_per_epoch=args.steps, eval_every=args.eval_every, 
                                   batch_size=args.batch_size, gamma=args.gamma, clip_ratio=args.clip_ratio, 
                                   lr=args.lr, lr_f=args.lr_f, ent_coeff=args.ent_coeff, 
                                   vf_coeff=args.vf_coeff, max_grad_norm=args.max_grad_norm, 
                                   clip_grad=args.clip_grad, train_iters=args.train_iters, 
                                   lam=args.lam, target_kl=args.target_kl, seq_len=args.seq_len, 
                                   seq_prefix=args.seq_prefix, seq_stride=args.seq_stride, log_dir=log_dir, 
                                   save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
    elif algo == 'td3':
        trainer = CDESP_TD3Trainer(rew_icm_max=args.rew_icm_max, fwd_coeff=args.fwd_coeff, lr_icm=args.lr_icm,
                                   icm_kwargs=icm_kwargs, env_fn=env_fn, wrappers_kwargs=wrappers_kwargs, 
                                   use_gpu=args.use_gpu, model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
                                   seed=args.seed, steps_per_epoch=args.steps, buf_size=args.buf_size, 
                                   gamma=args.gamma, polyak=args.polyak, lr=args.lr, lr_f=args.lr_f, 
                                   max_grad_norm=args.max_grad_norm, clip_grad=args.clip_grad, 
                                   batch_size=args.batch_size, start_steps=args.start_steps, 
                                   learning_starts=args.learning_starts, update_every=args.update_every,
                                   num_updates=args.num_updates, target_noise=args.target_noise, 
                                   noise_clip=args.noise_clip, policy_delay=args.policy_delay, 
                                   num_test_episodes=args.num_test_episodes, seq_len=args.seq_len, 
                                   seq_prefix=args.seq_prefix, seq_stride=args.seq_stride, 
                                   log_dir=log_dir, save_freq=args.save_freq, 
                                   checkpoint_freq=args.checkpoint_freq)
    trainer.train_mod(args.epochs)