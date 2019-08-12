# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:46:52 2019

@author: Andrei
"""
import torch as th
import numpy as np

from agent import DDPGAgent
from ma_per import MASimpleReplayBuffer

class MADDPGEngine():
  def __init__(self, action_size, state_size, n_agents, 
               GAMMA=0.99,
               MEMORY_SIZE=int(1e5),
               BATCH_SIZE=int(1e3),
               WARMP_UP=int(1e4),
               TAU=0.02,
               noise_scaling_factor=2.,
               noise_scaling_factor_dec=0.9999,
               dev=None):
    self.__name__ = 'MADDPG'
    self.n_agents = n_agents
    self.action_size = action_size
    self.GAMMA = GAMMA
    self.TAU = 0.02
    self.BATCH_SIZE = BATCH_SIZE
    self.WARM_UP = WARMP_UP
    self.state_size = state_size
    if dev is None:
      dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    self.dev = dev

    self.memory = MASimpleReplayBuffer(capacity=MEMORY_SIZE, 
                                       nr_agents=self.n_agents,
                                       engine='torch', 
                                       device=self.dev)

    self.agents = []
    for i in range(n_agents):
      self.agents.append(DDPGAgent(a_size=self.action_size, 
                                   s_size=self.state_size,
                                   dev=self.dev, 
                                   n_agents=self.n_agents,
                                   TAU=self.TAU))
    
    self.noise_scaling_factor = noise_scaling_factor
    self.noise_scaling_factor_dec = noise_scaling_factor_dec
    self.n_updates = 0
    return
  
  
  def act(self, states, add_noise):
    actions = np.zeros((self.n_agents, self.action_size))
    for i_agent in range(self.n_agents):
      if add_noise:
        nsf = self.noise_scaling_factor
        actions[i_agent] = self.agents[i_agent].act(states[i_agent], noise_scale=nsf)
      else:
        actions[i_agent] = self.agents[i_agent].act(states[i_agent], noise_scale=0.)
    if add_noise:
      self.noise_scaling_factor = self.noise_scaling_factor * self.noise_scaling_factor_dec
      self.noise_scaling_factor = max(0.5, self.noise_scaling_factor)        
    return actions
  
  
  def __step(self, states, actions, rewards, next_states, dones):
    self.add_exp(states, actions, rewards, next_states, dones)
    if len(self.memory) > self.WARM_UP:
      for i_agent in range(self.n_agents):
        self.train(agent_no=i_agent)
    return
      
  
  def train(self, n_samples=None):
    avail_data = len(self.memory)
    if avail_data < self.WARM_UP:
      return    
    for i_agent in range(self.n_agents):
      self._train(i_agent, n_samples=n_samples)
    self.update_targets()
    return
  
  def _train(self, agent_no, n_samples=None):
    """
    """
    if n_samples is None:
      n_samples = self.BATCH_SIZE
    # get n_samples from memory
    samples = self.memory.sample(n_samples)
    # prpare tensors for each agent
    all_exp = [x.transpose(0,1) for x in samples]

    experiences_per_agent = []
    for i in range(self.n_agents):
      experiences_per_agent.append([x[i] for x in all_exp])
    
    # these are the actual tensors for the current agent that is being trained
    (th_agent_obs, th_agent_action, th_agent_reward, 
     th_agent_next_obs, th_agent_done) = experiences_per_agent[agent_no]
    
    th_agent_reward = th_agent_reward.unsqueeze(1)
    th_agent_done = th_agent_done.unsqueeze(1)
    
    curr_agent = self.agents[agent_no]

    # first we generate next actions for all agents
    all_next_actions = []
    all_next_obs = []
    all_actions = []
    all_states = []
    for i in range(self.n_agents):
      # get observation for each agent perspective
      th_obs, th_action, th_reward, th_next_obs, th_done = experiences_per_agent[i]
      # now ask agent target actor to evaluate next action
      th_next_actions = self.agents[i].actor_target(th_next_obs)
      # prepare all actions buffer
      all_next_actions.append(th_next_actions)
      # we do not have a 'overview' observation so we concatenate all observations
      all_next_obs.append(th_next_obs)
      all_states.append(th_obs)
      all_actions.append(th_action)
      
    th_all_next_actions = th.cat(all_next_actions, dim=1)
    th_all_next_states = th.cat(all_next_obs, dim=1)
    th_all_actions = th.cat(all_actions, dim=1)
    th_all_states = th.cat(all_states, dim=1)
    
    
    # now are ca compute for our agent the actual Q value
    with th.no_grad():
      th_Q_target = curr_agent.critic_target(state=th_all_next_states,
                                             action=th_all_next_actions)
    
    th_y = th_agent_reward + self.GAMMA * th_Q_target * (1 - th_agent_done)
    th_y = th_y.detach()
    
    # now we train the agent critic
    
    th_q_value = curr_agent.critic(state=th_all_states, action=th_all_actions)
    loss_fn = th.nn.SmoothL1Loss()
    
    curr_agent.critic_optimizer.zero_grad()
    th_critic_loss = loss_fn(th_q_value, th_y)
    th_critic_loss = th_critic_loss.mean()
    th_critic_loss.backward()
    curr_agent.critic_optimizer.step()
    
    
    # now we prepare the input for actor optimization
    actors_actions = []
    for i in range(self.n_agents):
      th_obs, th_action, th_reward, th_next_obs, th_done = experiences_per_agent[i]
      th_actions = self.agents[i].actor(th_obs)
      if i != agent_no:
        th_actions = th_actions.detach()
      actors_actions.append(th_actions)
    th_actors_actions = th.cat(actors_actions, dim=1)
    
    curr_agent.actor_optimizer.zero_grad()
    th_actor_loss = -curr_agent.critic(state=th_all_states, action=th_actors_actions)
    th_actor_loss = th_actor_loss.mean()
    th_actor_loss.backward()
    curr_agent.actor_optimizer.step()
    
    return
  
  
  def update_targets(self):
    self.n_updates += 1
    for i in range(self.n_agents):
      self.agents[i].update_targets()
    return
  
  def add_exp(self, states, actions, rewards, next_states, dones):
    self.memory.add(states, actions, rewards, next_states, dones)
    return
      
    
    
      
    
  