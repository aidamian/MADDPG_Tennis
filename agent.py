# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:56:45 2019

@author: Andrei
"""
import numpy as np
import torch as th

from ma_per import MASimpleReplayBuffer

from brain import MADDPGActor, MADDPGCritic, layers_stats

from OUNoise import OUNoise

from time import time


def calc_huber_weighted(th_y_pred, th_y_true, d=1, th_weights=None):
  th_res = th_y_pred - th_y_true
  th_batch_loss1 = (th_res.abs()  <1).float() * (0.5 * (th_res**2))
  th_batch_loss2 = (th_res.abs() >=1).float() * (d * th_res.abs() - 0.5 * d)
  th_batch_loss = th_batch_loss1 + th_batch_loss2
  if th_weights is not None:
    th_weighted_batch_loss = th_weights * th_batch_loss 
  else:
    th_weighted_batch_loss = th_batch_loss
  th_weighted_loss = th_weighted_batch_loss.mean()
  return th_weighted_loss  

def calc_huber_weighted_residual(th_res, d=1, th_weights=None):
  th_batch_loss1 = (th_res.abs()  <1).float() * (0.5 * (th_res**2))
  th_batch_loss2 = (th_res.abs() >=1).float() * (d * th_res.abs() - 0.5 * d)
  th_batch_loss = th_batch_loss1 + th_batch_loss2
  if th_weights is not None:
    th_weighted_batch_loss = th_weights * th_batch_loss 
  else:
    th_weighted_batch_loss = th_batch_loss
  th_weighted_loss = th_weighted_batch_loss.mean()
  return th_weighted_loss  



class DDPGAgent():
  def __init__(self, a_size, s_size, dev, n_agents, overview_state=False, 
               LR=0.02, TAU=0.02):
    self.act_size = a_size
    self.obs_size = s_size
    self.dev = dev
    self.LR = LR
    self.TAU = TAU
    self.n_agents = n_agents
    self.overview_state = overview_state

    self.actor = MADDPGActor(input_size=self.obs_size,
                             output_size=self.act_size).to(self.dev)
    self.actor_target = MADDPGActor(input_size=self.obs_size,
                                    output_size=self.act_size).to(self.dev)
    self.actor_optimizer = th.optim.Adam(params=self.actor.parameters(),
                                         lr=self.LR)
    self._hard_update(local=self.actor, target=self.actor_target)
        
    obs_mult = 1 if overview_state else n_agents
    self.critic = MADDPGCritic(state_size=self.obs_size*obs_mult,
                               act_size=self.act_size*n_agents).to(self.dev)
    self.critic_target = MADDPGCritic(state_size=self.obs_size*obs_mult,
                                      act_size=self.act_size*n_agents).to(self.dev)
    self.critic_optimizer = th.optim.Adam(params=self.critic.parameters(),
                                          lr=self.LR)
    self._hard_update(local=self.critic, target=self.critic_target)     
    
    self.noise = OUNoise(a_size, scale=1.0)
    
    return
  
  
  
  def act(self, state, noise_scale):
    obs = th.tensor(state, dtype=th.float).to(self.dev)
    self.actor.eval()
    action = self.actor(obs).detach().cpu().numpy()
    self.actor.train()
    np_noise = noise_scale * self.noise.noise()
    action = action + np_noise
    action = np.clip(action, -1, 1)
    return action

  
  def update_targets(self):
    self.soft_update_actor()
    self.soft_update_critic()
    return


  def save(self, label):
    fn = '{}_actor_it_{:010}_{}.policy'.format(self.name, self.train_iters, label)
    th.save(self.actor.state_dict(), fn)
    return

  
  def soft_update_actor(self):
    self._soft_update(self.actor, self.actor_target, self.TAU)
    return

      
  def soft_update_critic(self):
    self._soft_update(self.critic, self.critic_target, self.TAU)
    return
      
  
  def _hard_update(self, local, target):
    target.load_state_dict(local.state_dict())
    return
      
  def _soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)    
    return
  
  def debug_weights(self):
    layers_stats(self.actor)
    layers_stats(self.critic)
    return
  