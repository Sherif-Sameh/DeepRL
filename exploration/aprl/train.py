import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from core.sac.train import SACTrainer
from core.sac.train import get_parser as get_parser_sac
from core.rl_utils import call_env_method

from exploration.aprl.models.mlp import RegSACMLPActorCritic, MLPDynamicsModel
from exploration.aprl.regact_wrapper import RegularizeAction

class DynamicsTrainer:
    def __init__(self, env, **dyn_mod_kwargs):
        # Initialize dynamics model
        self.dyn_mod = MLPDynamicsModel(env, **dyn_mod_kwargs)
        self.dyn_mod.layer_summary()
        
    def _calc_dyn_loss(self, obs: torch.Tensor, act: torch.Tensor, 
                       obs_next: torch.Tensor, loss_mask: torch.Tensor):
        obs_pred = self.dyn_mod.forward(obs, act)

        return F.mse_loss(obs_pred[loss_mask], obs_next[loss_mask])

class APRL_SACTrainer(SACTrainer, DynamicsTrainer):
    """ 
    Adaptive Policy ReguLarization (APRL)

    :param growth_rate: Rate (in number of iterations) at which the actions current 
        upper limit is grown. Actions limit reaches act_end after growth_rate 
        iterations with without invalidating the dynamic error threshold.    
    :param shrink_rate: Rate at which the lower threshold for the actions current 
        upper limit is shrunk at every time the dynamic error threshold is exceeded.
    :param act_start: The initial lower threshold for the actions limit. 
        This limit gets updated during training. 
    :param act_end: The upper threshold for the actions limit. Never updated.
    :param dyn_threshold: The forward dynamics error threshold for triggering the 
        shrinking of the action limit's lower threshold. 
    :param lr: Learning rate used by the optimizer and shared between all of the 
        actor, critics and the forward dynamics model. 
    :param replay_ratio: Number of parameter updates to the critics per a single 
        parameter update of the actor and the dynamics model.  
    :param max_grad_steps: Maximum number of successive gradient steps uninterrupted 
        by any triggers of the dynamics error threshold, after which the parameters 
        of the actor, critics and dynamics model are reset.  
    :param penalty_weight: Weight factor applied to the actor's added loss component 
        for exceeding the set action limits. 
    :param dyn_mod_kwargs: A dictionary of key-value arguments to pass to the dynamics
        model contructor alongside a reference to the environment.  
    """
    def __init__(self, env_fn, growth_rate=10000, shrink_rate=0.9, act_start=0.3, 
                 act_end=0.7, dyn_threshold=1.5, lr=3e-4, replay_ratio=5, 
                 max_grad_steps=40000, penalty_weight=10.0, dyn_mod_kwargs=dict(), 
                 **sac_kwargs):
        self.replay_ratio = replay_ratio
        self.max_grad_steps = max_grad_steps
        self.penalty_weight = penalty_weight

        # Initialize actor-critic trainer
        SACTrainer.__init__(self, env_fn, lr=lr, start_steps=0, update_every=1, 
                            num_updates=len(env_fn), **sac_kwargs)
        
        # Initialize dynamics model trainer and dynamics model
        DynamicsTrainer.__init__(self, self.env, **dyn_mod_kwargs)
        self.dyn_mod.to(self.device)

        # Update optimizer to include parameters of dynamics model
        self.ac_optim = Adam([
            {'params': self.ac_mod.parameters()},
            {'params': self.dyn_mod.parameters()}
        ], lr=lr)

        # Wrap environment with action regularization wrapper
        self.env = RegularizeAction(self.env, self.device, growth_rate, shrink_rate, 
                                    act_start, act_end, dyn_threshold, self.dyn_mod)
    
    def _calc_pi_loss(self, alpha, obs, loss_mask):
        # Normal SAC actor loss
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        act, log_prob, pre_act = self.ac_mod.actor.log_prob(obs)  

        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_1 = self.ac_mod.critic_1.forward_actions(obs, act)
        self.ac_mod.reset_hidden_states(self.device, batch_size=obs.shape[0])
        q_vals_2 = self.ac_mod.critic_2.forward_actions(obs, act)
        q_vals = torch.min(q_vals_1, q_vals_2)
        q_vals.squeeze_(dim=-1)        
        pi_loss = (-q_vals[loss_mask] + alpha * log_prob[loss_mask]).mean()

        # Additional loss to penalize actions > dynamic action limit
        action_limit = call_env_method(self.env, "get_action_limit").item()
        act_abs = torch.abs(act[loss_mask])
        action_out = torch.where(act_abs > action_limit, act_abs - action_limit, 
                                 torch.zeros_like(act_abs))
        action_penalty = torch.sum(action_out, dim=-1).mean()

        return pi_loss + self.penalty_weight * action_penalty, log_prob
    
    def _update_params(self, epoch):
        def get_batch():
            to_tensor = lambda np_arr, dtype: torch.as_tensor(np_arr, dtype=dtype, device=self.device)
            obs, act, rew, obs_next, done, mask = self.buf.get_batch()
            if self.norm_obs:
                obs, obs_next = self._normalize_obs(obs), self._normalize_obs(obs_next)
            if self.norm_rew:
                rew = self._normalize_rew(rew)

            return to_tensor(obs, torch.float32), to_tensor(act, torch.float32), \
                to_tensor(rew, torch.float32), to_tensor(obs_next, torch.float32), \
                to_tensor(done, torch.bool), to_tensor(mask, torch.bool)
        
        # Update critics seperately according to set replay ratio
        alpha_det = self.alpha_mod.forward().detach().to(self.device)
        losses_critic = torch.zeros(self.replay_ratio, 2)
        for i in range(self.replay_ratio):
            obs, act, rew, obs_next, done, mask = get_batch()
            self.ac_optim.zero_grad()
            loss_q1, loss_q2 = self._calc_q_loss(alpha_det, obs, act, rew, 
                                                 obs_next, done, mask)
            losses_critic[i, 0], losses_critic[i, 1] = loss_q1.item(), loss_q2.item()
            loss = loss_q1 + loss_q2
            loss.backward()
            self.ac_optim.step()
            
            # Update target critic networks
            self.ac_mod.critic_1.update_target(self.polyak)
            self.ac_mod.critic_2.update_target(self.polyak)

        # Update actor and dynamics model
        obs, act, rew, obs_next, done, mask = get_batch()
        self.ac_optim.zero_grad()
        loss_dyn = self._calc_dyn_loss(obs, act, obs_next, mask)
        loss_dyn.backward()

        self.ac_mod.critic_1.set_grad_tracking(val=False)    
        self.ac_mod.critic_2.set_grad_tracking(val=False)    
        loss_pi, logp = self._calc_pi_loss(alpha_det, obs, mask)
        loss_pi.backward()
        self.ac_mod.critic_1.set_grad_tracking(val=True)    
        self.ac_mod.critic_2.set_grad_tracking(val=True)
        self.ac_optim.step()

        # Update temperature coefficient if it's estimated online
        if self.auto_alpha == True:    
            self.alpha_optim.zero_grad()
            loss_alpha = self._calc_alpha_loss(logp, mask)
            loss_alpha.backward()
            self.alpha_optim.step()
            if self.alpha_min > 0:    
                with torch.no_grad(): self.alpha_mod.log_alpha.clamp_min_(torch.tensor(np.log(self.alpha_min)))
            self.writer.add_scalar('Loss/LossAlpha', loss_alpha.item(), epoch+1)

        # Log training statistics
        self.writer.add_scalar('Loss/LossQ1', losses_critic[:, 0].mean(), epoch+1)
        self.writer.add_scalar('Loss/LossQ2', losses_critic[:, 1].mean(), epoch+1)
        self.writer.add_scalar('Loss/LossPi', loss_pi.item(), epoch+1)
        self.writer.add_scalar('Loss/LossDyn', loss_dyn.item(), epoch+1)
    
    def _train(self, epoch):
        super()._train(epoch)

        # Reset weights of all networks if neccesary
        step_count = call_env_method(self.env, "get_step_count")
        if (step_count * self.replay_ratio) > self.max_grad_steps:
            self.ac_mod.reset_weights()
            self.dyn_mod.reset_weights()

    def _eval(self):
        # Disable action regualization for evaluation and re-enable it after
        call_env_method(self.env, "disable_act_reg")
        self.ac_mod.eval()
        super()._eval()
        self.ac_mod.train()
        call_env_method(self.env, "enable_act_reg")
    
    def _log_ep_stats(self, epoch, q_val_list):
        super()._log_ep_stats(epoch, q_val_list)
        action_limit = call_env_method(self.env, "get_action_limit")        
        self.writer.add_scalar('ActionLimit', action_limit, epoch+1)

def get_algo_type():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['sac'])

    # Parse first argument and pass on remaining ones
    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    return args.algo, remaining_args

if __name__ == '__main__':
    # Parse first argument to know algorithm type
    algo, remaining_args = get_algo_type()

    # Get parser for all remaining DRL algorithm arguments and add APRL arguments 
    if algo == 'sac':
        parser = get_parser_sac()
    
    # APRL parameters
    parser.add_argument('--hid_dyn', nargs='+', type=int, default=[64, 64])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    
    parser.add_argument('--growth_rate', type=int, default=10000)
    parser.add_argument('--shrink_rate', type=float, default=0.9)
    parser.add_argument('--act_start', type=float, default=0.3)
    parser.add_argument('--act_end', type=float, default=0.7)
    parser.add_argument('--dyn_threshold', type=float, default=1.5)
    parser.add_argument('--replay_ratio', type=int, default=5)
    parser.add_argument('--max_grad_steps', type=int, default=40000)
    parser.add_argument('--penalty_weight', type=float, default=10)

    args = parser.parse_args(remaining_args)
    args.exp_name = 'aprl_'+ args.exp_name
    assert args.policy == 'mlp', "APRL must be used with a MLP policy"

    # Set directory for logging
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_script_dir + '/../../runs/' + args.env + '/'
    log_dir += args.exp_name + '/' + args.exp_name + f'_s{args.seed}'

    # Setup actor-critic and dynamics model kwargs
    max_ep_len = args.max_ep_len if args.max_ep_len > 0 else None
    if algo == 'sac':
        ac = RegSACMLPActorCritic
        ac_kwargs = dict(hidden_sizes_actor=args.hid_act, 
                         hidden_sizes_critic=args.hid_cri,
                         hidden_acts_actor=torch.nn.ReLU, 
                         hidden_acts_critic=torch.nn.ReLU,
                         dropout_prob=args.dropout_prob)
        
    dyn_kwargs = dict(hidden_sizes=args.hid_dyn, 
                      hidden_acts=torch.nn.ReLU)
    
    # Setup environment lambda
    env_fn = [lambda render_mode=None: gym.make(args.env, max_episode_steps=max_ep_len, 
                                                render_mode=render_mode)] * args.cpu
    wrappers_kwargs = dict()
    
    # Setup the trainer and begin training
    if algo == 'sac':
        trainer = APRL_SACTrainer(env_fn, growth_rate=args.growth_rate, shrink_rate=args.shrink_rate, 
                                  act_start=args.act_start, act_end=args.act_end, 
                                  dyn_threshold=args.dyn_threshold, replay_ratio=args.replay_ratio, 
                                  max_grad_steps=args.max_grad_steps, penalty_weight=args.penalty_weight,
                                  dyn_mod_kwargs=dyn_kwargs, wrappers_kwargs=wrappers_kwargs, 
                                  use_gpu=args.use_gpu, model_path=args.model_path, ac=ac, 
                                  ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps, 
                                  buf_size=args.buf_size, gamma=args.gamma, polyak=args.polyak, lr=args.lr, 
                                  lr_f=args.lr_f, pre_act_coeff=args.pre_act_coeff, norm_rew=args.norm_rew, 
                                  norm_obs=args.norm_obs, max_grad_norm=args.max_grad_norm, 
                                  clip_grad=args.clip_grad, alpha=args.alpha, alpha_min=args.alpha_min, 
                                  entropy_target=args.entropy_target, auto_alpha=args.auto_alpha, 
                                  batch_size=args.batch_size, learning_starts=args.learning_starts, 
                                  num_test_episodes=args.num_test_episodes, seq_len=args.seq_len, 
                                  seq_prefix=args.seq_prefix, seq_stride=args.seq_stride, log_dir=log_dir, 
                                  save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
        
    trainer.learn(args.epochs, ep_init=args.ep_init)