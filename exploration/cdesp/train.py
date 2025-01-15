import os
import sys
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from core.ppo.train import PPOTrainer
from core.ppo.train import get_parser as get_parser_ppo
from core.td3.train import TD3Trainer
from core.td3.train import get_parser as get_parser_td3
from core.ppo.models.cnn_lstm import CNNLSTMActorCritic as PPOActorCritic
from core.td3.models.cnn_lstm import CNNLSTMActorCritic as TD3ActorCritic
from core.rl_utils import SkipAndScaleObservation

from exploration.cdesp.models.icm import IntrinsicCuriosityModule
from exploration.cdesp.icm_wrapper import IntrinsicRewardWrapper
from exploration.utils import is_vizdoom_env
from exploration.rl_utils import VizdoomToGymnasium

class CDESPTrainer:
    def __init__(self, env, fwd_coeff, target_window, auto_beta, explor_coeff, 
                 explor_coeff_f, max_grad_norm_icm, lr_icm, icm_kwargs=dict()):
        self.fwd_coeff = fwd_coeff
        self.ret_ext = [0] * target_window
        self.ret_ext_min = 0
        self.auto_beta = auto_beta
        self.explor_coeff = explor_coeff
        self.explor_coeff_diff = explor_coeff_f - explor_coeff
        self.max_grad_norm_icm = max_grad_norm_icm
        
        # Initialize ICM module
        self.icm_mod = IntrinsicCuriosityModule(env, **icm_kwargs)
        self.icm_mod.layer_summary()

        # Initialize SGD optimizer for ICM's parameters 
        self.icm_optim = Adam([
            {'params': self.icm_mod.inv_mod.parameters()},
            {'params': self.icm_mod.fwd_mod.parameters()},
            {'params': self.icm_mod.log_beta_multiplier, 'lr': 1e-2}
        ], lr=lr_icm)
        
        # Determine inverse model loss function based on environment action space type
        if isinstance(env.single_action_space, Discrete):
            self.inv_loss = self._calc_inv_loss_discrete
        elif isinstance(env.single_action_space, Box):
            self.inv_loss = self._calc_inv_loss_cont

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
                          features_target[loss_mask_icm]) * features.shape[-1]
    
    def _calc_beta_loss(self, beta_0: torch.Tensor, rew_mean: torch.Tensor, 
                        target_rew_mean: torch.Tensor):
        beta = self.icm_mod.get_beta().detach()

        return (self.icm_mod.log_beta_multiplier * ((beta/beta_0) * rew_mean - target_rew_mean))
    
    def _step_icm_params(self, obs: torch.Tensor, act: torch.Tensor, 
                         mask: torch.Tensor):    
        # Get inverse and forward model losses
        self.icm_optim.zero_grad()
        loss_inv, features = self.inv_loss(obs, act, mask)
        loss_fwd = self._calc_fwd_loss(features, act, mask)
        loss_fwd = torch.tensor(0) if torch.isnan(loss_fwd) else loss_fwd

        # Combine losses and update parameters
        loss_icm = loss_inv + self.fwd_coeff * loss_fwd
        loss_icm.backward()
        torch.nn.utils.clip_grad_norm_(self.icm_mod.fwd_mod.parameters(), 
                                       max_norm=self.max_grad_norm_icm)
        self.icm_optim.step()

        return loss_inv.item(), loss_fwd.item()

    def _step_icm_beta(self, beta_0: torch.Tensor, ret_mean: torch.Tensor, 
                       ret_mean_ext: torch.Tensor):
        target_rew_mean = self.explor_coeff * ret_mean_ext
        if (self.auto_beta == True) and (target_rew_mean > 0):
            self.icm_optim.zero_grad()
            loss_beta = self._calc_beta_loss(beta_0, ret_mean, target_rew_mean)
            loss_beta.backward()
            self.icm_optim.step()
        else:
            loss_beta = torch.tensor(0)
        
        return loss_beta.item()
    
    def _step_exploration_coeff(self, epochs_total):
        self.explor_coeff += (self.explor_coeff_diff/(epochs_total+1))

class CDESP_PPOTrainer(CDESPTrainer, PPOTrainer):
    def __init__(self, fwd_coeff=0.25, target_window=10, auto_beta=True, explor_coeff=1.0, 
                 explor_coeff_f=0.0, rew_icm_max=1.0, max_grad_norm_icm=0.5, lr_icm=1e-3, 
                 icm_kwargs=dict(), **ppo_args):
        # Initialize actor-critic trainer
        PPOTrainer.__init__(self, **ppo_args)
        
        # Initialize CDESP trainer and ICM module
        CDESPTrainer.__init__(self, self.env, fwd_coeff, target_window, auto_beta, 
                              explor_coeff, explor_coeff_f, max_grad_norm_icm, lr_icm, 
                              icm_kwargs)
        self.icm_mod.to(self.device)

        # Wrap environment with ICM reward wrapper
        self.env = IntrinsicRewardWrapper(rew_icm_max, self.env, self.icm_mod, self.device)
        self.ep_ret_icm_list = []
    
    def _train(self, epoch):
        # Update actor-critic's parameters
        super()._train(epoch)

        # Calculate extrinsic and current intrinsic returns
        self.ret_ext.append(np.mean(self.ep_ret_list)); self.ret_ext.pop(0)
        self.ret_ext_min = min(self.ret_ext_min, self.ret_ext[-1])
        ret_mean_ext = torch.tensor(np.mean(self.ret_ext) - self.ret_ext_min)
        ret_mean = torch.tensor(np.mean(self.ep_ret_icm_list))
        
        # Update ICM's parameters
        beta_0 = self.icm_mod.get_beta().detach()
        dataloader = self.buf.get_dataloader(self.device)
        for obs, act, _, _, _, mask in dataloader:
            loss_inv, loss_fwd = self._step_icm_params(obs, act, mask)
            loss_beta = self._step_icm_beta(beta_0, ret_mean, ret_mean_ext)
        self._step_exploration_coeff(self.epochs)
                 
        # Log epoch statistics for ICM losses
        self.writer.add_scalar('Loss/LossInv', loss_inv, epoch+1)
        self.writer.add_scalar('Loss/LossFwd', loss_fwd, epoch+1)
        self.writer.add_scalar('Loss/LossBeta', loss_beta, epoch+1)

    def _proc_env_rollout(self, env_id, val_terminal, rollout_len, ep_ret, ep_len):
        super()._proc_env_rollout(env_id, val_terminal, rollout_len, ep_ret, ep_len)
        
        # Get and store sum of intrinsic rewards from wrapper
        ep_ret_icm, ep_ret = self.env.env.get_and_clear_return(env_id)
        self.ep_ret_icm_list.append(ep_ret_icm)
        
        # Replace combined return with extrinsic return
        self.ep_ret_list[-1] = ep_ret

    def _log_ep_stats(self, epoch):
        super()._log_ep_stats(epoch)
        ep_ret_icm_np = np.array(self.ep_ret_icm_list)
        self.writer.add_scalar('EpRetICM/mean', ep_ret_icm_np.mean(), epoch+1)
        self.ep_ret_icm_list = []

class CDESP_TD3Trainer(CDESPTrainer, TD3Trainer):
    def __init__(self, fwd_coeff=0.25, target_window=10, auto_beta=True, explor_coeff=1.0, 
                 explor_coeff_f=0.0, rew_icm_max=1.0, max_grad_norm_icm=0.5, lr_icm=1e-3, 
                 icm_kwargs=dict(), **td3_args):
        # Initialize actor-critic trainer
        TD3Trainer.__init__(self, **td3_args)
        
        # Initialize CDESP trainer and ICM module
        CDESPTrainer.__init__(self, self.env, fwd_coeff, target_window, auto_beta, 
                              explor_coeff, explor_coeff_f, max_grad_norm_icm, lr_icm, 
                              icm_kwargs)
        self.icm_mod.to(self.device)

        # Wrap environment with ICM reward wrapper
        self.env = IntrinsicRewardWrapper(rew_icm_max, self.env, self.icm_mod, self.device)
        self.ep_ret_icm_list = []
    
    def _train(self, epoch):
        # Update actor-critic's parameters
        super()._train(epoch)

        # Calculate extrinsic and current intrinsic returns
        self.ret_ext.append(np.mean(self.ep_ret_list)); self.ret_ext.pop(0)
        self.ret_ext_min = min(self.ret_ext_min, self.ret_ext[-1])
        ret_mean_ext = torch.tensor(np.mean(self.ret_ext) - self.ret_ext_min)
        ret_mean = torch.tensor(np.mean(self.ep_ret_icm_list))
                
        # Update ICM's parameters
        beta_0 = self.icm_mod.get_beta().detach()
        obs, act, _, _, _, mask = self.buf.get_batch(self.device)
        loss_inv, loss_fwd = self._step_icm_params(obs, act, mask)
        loss_beta = self._step_icm_beta(beta_0, ret_mean, ret_mean_ext)
        self._step_exploration_coeff(self.epochs)

        # Log epoch statistics for ICM losses
        self.writer.add_scalar('Loss/LossInv', loss_inv, epoch+1)
        self.writer.add_scalar('Loss/LossFwd', loss_fwd, epoch+1)
        self.writer.add_scalar('Loss/LossBeta', loss_beta, epoch+1)

    def _proc_env_rollout(self, env_id, ep_len):
        super()._proc_env_rollout(env_id, ep_len)

        # Get and store sum of intrinsic rewards from wrapper
        ep_ret_icm, _ = self.env.env.get_and_clear_return(env_id)
        self.ep_ret_icm_list.append(ep_ret_icm)

    def _eval(self):
        # Disable intrinsic rewards for evaluation and re-enable them after
        self.env.env.disable_intrinsic_reward()
        super()._eval()
        self.env.env.enable_intrinsic_reward()
    
    def _log_ep_stats(self, epoch, q_val_list):
        super()._log_ep_stats(epoch, q_val_list)
        ep_ret_icm_np = np.array(self.ep_ret_icm_list)
        self.writer.add_scalar('EpRetICM/mean', ep_ret_icm_np.mean(), epoch+1)
        self.ep_ret_icm_list = []

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
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--auto_beta', action="store_true", default=False)
    parser.add_argument('--hid_inv', nargs='+', type=int, default=[256])
    parser.add_argument('--hid_fwd', nargs='+', type=int, default=[256])

    # Rest of training parameters
    parser.add_argument('--fwd_coeff', type=float, default=0.25)
    parser.add_argument('--target_window', type=int, default=10)
    parser.add_argument('--explor_coeff', type=float, default=2.0)
    parser.add_argument('--explor_coeff_f', type=float, default=0)
    parser.add_argument('--rew_icm_max', type=float, default=None)
    parser.add_argument('--max_grad_norm_icm', type=float, default=0.5)
    parser.add_argument('--lr_icm', type=float, default=1e-3)

    # Miniworld specific arguments
    parser.add_argument('--vizdoom_obs_size', type=int, default=42)

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
    if is_vizdoom_env(args.env):
        env_fn_def = lambda render_mode=None: VizdoomToGymnasium(
            gym.make(args.env, render_mode=render_mode, max_episode_steps=max_ep_len),
            img_size=args.vizdoom_obs_size)
    else:
        env_fn_def = lambda render_mode=None: gym.make(args.env, render_mode=render_mode, 
                                                       max_episode_steps=max_ep_len)
    env_fn = [lambda render_mode=None: FrameStackObservation(SkipAndScaleObservation(
        GrayscaleObservation(env_fn_def(render_mode=render_mode)), skip=args.action_rep), 
        stack_size=args.in_channels)] * args.cpu
    wrappers_kwargs = {
        'SkipAndScaleObservation': {'skip': args.action_rep},
        'FrameStackObservation': {'stack_size': args.in_channels}
    }
    if is_vizdoom_env(args.env): 
        wrappers_kwargs['VizdoomToGymnasium'] = {'img_size': args.vizdoom_obs_size}

    # Setup the trainer and begin training
    if algo == 'ppo':
        trainer = CDESP_PPOTrainer(fwd_coeff=args.fwd_coeff, target_window=args.target_window, 
                                   auto_beta=args.auto_beta, explor_coeff=args.explor_coeff, 
                                   explor_coeff_f=args.explor_coeff_f, rew_icm_max=args.rew_icm_max, 
                                   max_grad_norm_icm=args.max_grad_norm_icm, lr_icm=args.lr_icm, 
                                   icm_kwargs=icm_kwargs, env_fn=env_fn, 
                                   wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                                   model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
                                   seed=args.seed, steps_per_epoch=args.steps, eval_every=args.eval_every, 
                                   batch_size=args.batch_size, gamma=args.gamma, clip_ratio=args.clip_ratio, 
                                   lr=args.lr, lr_f=args.lr_f, ent_coeff=args.ent_coeff, 
                                   vf_coeff=args.vf_coeff, max_grad_norm=args.max_grad_norm, 
                                   clip_grad=args.clip_grad, train_iters=args.train_iters, 
                                   lam=args.lam, target_kl=args.target_kl, seq_len=args.seq_len, 
                                   seq_prefix=args.seq_prefix, seq_stride=args.seq_stride, log_dir=log_dir, 
                                   save_freq=args.save_freq, checkpoint_freq=args.checkpoint_freq)
    elif algo == 'td3':
        trainer = CDESP_TD3Trainer(fwd_coeff=args.fwd_coeff, target_window=args.target_window,
                                   auto_beta=args.auto_beta, explor_coeff=args.explor_coeff, 
                                   explor_coeff_f=args.explor_coeff_f, rew_icm_max=args.rew_icm_max, 
                                   max_grad_norm_icm=args.max_grad_norm_icm, lr_icm=args.lr_icm, 
                                   icm_kwargs=icm_kwargs, env_fn=env_fn, 
                                   wrappers_kwargs=wrappers_kwargs, use_gpu=args.use_gpu, 
                                   model_path=args.model_path, ac=ac, ac_kwargs=ac_kwargs, 
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
        
    trainer.learn(args.epochs, ep_init=args.ep_init)