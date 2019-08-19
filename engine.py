# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:46:52 2019

@author: Andrei
"""
import torch as th
import numpy as np

from agent import DDPGAgent
from ma_per import MASimpleReplayBuffer, MASimplePER
from brain import layers_stats

from collections import deque

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


class MADDPGEngine():
  def __init__(self, action_size, state_size, n_agents, 
               GAMMA=0.99,
               MEMORY_SIZE=int(2e4),
               BATCH_SIZE=256,
               WARMP_UP=4096,
               TAU=5e-3,
               actor_layers=[256,128],
               actor_input_bn=True,
               actor_hidden_bn=False,
               critic_state_layers=[256],
               critic_final_layers=[256, 128],
               critic_state_bn=True,
               critic_final_bn=False,
               apply_post_bn=False,
               noise_scaling_factor=2.,
               noise_scaling_factor_dec=0.9,
               noise_scaling_min=0.2,
               LR_ACTOR=1e-4,
               LR_CRITIC=2e-4,
               huber_loss=False,
               DEBUG=False,
               OUNoise=True,
               activation='relu',
               PER=None,
               min_non_zero_prc=0.35,
               name='MADDPG',
               dev=None):
    self.__name__ = name
    self.DEBUG = DEBUG
    self.n_agents = n_agents
    self.action_size = action_size
    self.GAMMA = GAMMA
    self.TAU = 0.02
    self.BATCH_SIZE = BATCH_SIZE
    self.WARM_UP = WARMP_UP
    self.state_size = state_size
    self.current_episode = 0
    if dev is None:
      dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    self.dev = dev
    self.huber_loss = huber_loss
    
    if PER == 'none':
      self.min_non_zero_prc = 0
      self.PER = False
      self.memory = MASimpleReplayBuffer(capacity=MEMORY_SIZE, 
                                         nr_agents=self.n_agents,
                                         engine='torch', 
                                         device=self.dev,
                                         min_non_zero_prc=0)
    elif PER == "sparsity":
      self.min_non_zero_prc = min_non_zero_prc
      self.PER = False
      self.memory = MASimpleReplayBuffer(capacity=MEMORY_SIZE, 
                                         nr_agents=self.n_agents,
                                         engine='torch', 
                                         device=self.dev,
                                         min_non_zero_prc=min_non_zero_prc)
    elif PER == 'PER1':
      self.PER = True
      self.memory = MASimplePER(capacity=MEMORY_SIZE, 
                                nr_agents=self.n_agents,
                                engine='torch', 
                                device=self.dev,
                                )
    else:
      raise ValueError("Unknown PER parameter value '{}'".format(PER))

    self.agents = []
    for i in range(n_agents):
      self.agents.append(DDPGAgent(a_size=self.action_size, 
                                   s_size=self.state_size,
                                   dev=self.dev, 
                                   n_agents=self.n_agents,
                                   TAU=self.TAU,
                                   bn_post=apply_post_bn,
                                   actor_layers=actor_layers,
                                   actor_input_bn=actor_input_bn,
                                   actor_hidden_bn=actor_hidden_bn,
                                   critic_state_bn=critic_state_bn,
                                   critic_final_bn=critic_final_bn,
                                   critic_final_layers=critic_final_layers,
                                   critic_state_layers=critic_state_layers,
                                   OUnoise=OUNoise,
                                   LR_ACTOR=LR_ACTOR,
                                   LR_CRITIC=LR_CRITIC,
                                   activation=activation,
                                   name='{}_Agent_{}'.format(
                                       self.__name__, i+1)))
      
    self.agents[0].show_architecture()
    
    self.noise_scaling_factor = noise_scaling_factor
    self.noise_scaling_factor_dec = noise_scaling_factor_dec
    self.n_updates = 0
    self.noise_scaling_min = noise_scaling_min
    return
  
    
  
  def act(self, states, add_noise):
    actions = np.zeros((self.n_agents, self.action_size))
    for i_agent in range(self.n_agents):
      if add_noise:
        nsf = self.noise_scaling_factor
        actions[i_agent] = self.agents[i_agent].act(states[i_agent], noise_scale=nsf)
      else:
        actions[i_agent] = self.agents[i_agent].act(states[i_agent], noise_scale=0.)
    return actions
  
  
  def __step(self, states, actions, rewards, next_states, dones):
    self.add_exp(states, actions, rewards, next_states, dones)
    if len(self.memory) > self.WARM_UP:
      for i_agent in range(self.n_agents):
        self.train(agent_no=i_agent)
    return
      
  
  def save(self):
    print("Saving agents actors...")
    for agent in self.agents:
      agent.train_iters = self.current_episode
      agent.save('')
    return
  
  def load(self, lst_agent_files):
    if len(lst_agent_files) != len(self.agents):
      raise ValueError("Number of agents must equal number of files!")
    print("{} Loading actors...".format(self.__name__))
    for i, agent in enumerate(self.agents):
      agent.load(lst_agent_files[i])
  
  
  def train(self, n_samples=None):
    avail_data = len(self.memory)
    if avail_data < self.WARM_UP:
      return    
    for i_agent in range(self.n_agents):
      self._train(i_agent, n_samples=n_samples)
    self.update_targets()

    self.noise_scaling_factor = self.noise_scaling_factor * self.noise_scaling_factor_dec
    self.noise_scaling_factor = max(self.noise_scaling_min, self.noise_scaling_factor)        
    return
  
  def _train(self, agent_no, n_samples=None):
    """
    """
    if n_samples is None:
      n_samples = self.BATCH_SIZE
      
    ### get n_samples from memory
    ### the following lines of code are based on non-vectorized approaches for readability
    ### and educational purpose
    if self.PER:
      # PER_weights are the Importance Sampling weights (inverse to the importance)
      samples, PER_indices, PER_weights = self.memory.sample(agent=agent_no, 
                                                             batch_size=n_samples)
    else:
      samples = self.memory.sample(agent=agent_no, n=n_samples)

    _states, _actions, _rewards, _next_states, _dones = samples
    # prpare tensors for each agent
    all_exp = [x.transpose(0,1) for x in samples]

    experiences_per_agent = []
    for i in range(self.n_agents):
      experiences_per_agent.append([x[i] for x in all_exp])
    
    # these are the actual tensors for the current agent that is being trained
    (th_agent_obs, th_agent_action, th_agent_reward, 
     th_agent_next_obs, th_agent_done) = experiences_per_agent[agent_no]
    
    ###
    ###
    ###
    
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
    
    
    # now we are computing the actual Q value of our current agent
    # the critic receives all state and all actions
    # but only the reward of our current agent so the critic analyzes only the 
    # current revenue stream and generates the current agent Q value
    with th.no_grad():
      th_Q_target = curr_agent.critic_target(state=th_all_next_states,
                                             action=th_all_next_actions)
    
    th_y = th_agent_reward + self.GAMMA * th_Q_target * (1 - th_agent_done)
    th_y = th_y.detach()
    
    # now we train the agent critic
    
    th_q_value = curr_agent.critic(state=th_all_states, action=th_all_actions)
        
    curr_agent.critic_optimizer.zero_grad()
    
    if self.PER:
      th_qfunc_residual = th_y - th_q_value
      np_errors = th.abs(th_qfunc_residual).cpu().detach().numpy()
      self.memory.batch_update(agent=agent_no, 
                               batch_indices=PER_indices, 
                               batch_priorities=np_errors)      
      if self.huber_loss:
        th_critic_loss = calc_huber_weighted_residual(th_res=th_qfunc_residual,
                                                      th_weights=PER_weights)
      else:
        th_critic_loss = th_qfunc_residual.pow(2)
        if PER_weights is not None:
          th_critic_loss *= PER_weights
        th_critic_loss = th_critic_loss.mean()    
    else:      
      if self.huber_loss:
        loss_fn = th.nn.SmoothL1Loss()
      else:
        loss_fn = th.nn.MSELoss()
      th_critic_loss = loss_fn(th_q_value, th_y)
      th_critic_loss = th_critic_loss.mean()
      
    th_critic_loss.backward()
    if self.DEBUG:
      if layers_stats(curr_agent.critic):
        print(self.current_episode)
    th.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
    curr_agent.critic_optimizer.step()
    curr_agent.critic_optimizer.zero_grad()  
    
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
    if self.DEBUG:
      if layers_stats(curr_agent.actor):
        print(self.current_episode)
    th.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 1)
    curr_agent.actor_optimizer.step()
    curr_agent.actor_optimizer.zero_grad()    
    return
  
  
  def maybe_sample(self, n_samples):
    # check if we have any good samples
    rewards = [x.reward for x in self.memory.memory]
    if max(rewards) <=0:
      return None
    done = False
    while not done:
      samples = self.memory.sample(n_samples)
      _, _ , th_rewards, _, _ = samples
      if th_rewards.max() > 0:
        break
    return samples
  
  
  def update_targets(self):
    self.n_updates += 1
    for i in range(self.n_agents):
      self.agents[i].update_targets()
    return
  
  def add_exp(self, states, actions, rewards, next_states, dones):
    self.memory.add(states, actions, rewards, next_states, dones)
    return
            
    
  def run_on_unity(self, env, brain_name, n_episodes=1000,
              n_ep_per_update=1, n_update_per_ep=3, time_steps_per_episode=10000,
              n_update_per_step=1, DEBUG=0):  
    last_scores = deque(maxlen=100)
    last_steps = deque(maxlen=100)
    all_scores = []
    all_avg_scores = []
    all_avg_steps = []
    solved = 0
    print("Unity run stats for MADDPG main training loop:")
    print("  Updates post episode:  {:.1f}".format(n_update_per_ep/n_ep_per_update))
    print("  Updates per step:      {}".format(n_update_per_step))
    print("  Max steps per episode: {}".format(time_steps_per_episode))
    print("  Max no episodes:       {}".format(n_episodes))
    
    if DEBUG > 0:
      n_episodes = DEBUG
      
    for i_episode in range(1,n_episodes+1):
      self.current_episode = i_episode
      env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
      states = env_info.vector_observations                  # get the current state (for each agent)
      ep_rewards = []
      for ts in range(time_steps_per_episode):
        actions = self.act(states, add_noise=True)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        self.add_exp(states, actions, rewards, next_states, dones)
        for updt in range(n_update_per_step):
          self.train()
        ep_rewards.append(rewards)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
          break
      last_steps.append(ts)
      np_rewards = np.array(ep_rewards)
      scores = np_rewards.sum(axis=0)
      last_score = scores.max()
      all_scores.append(last_score)
      last_scores.append(last_score)
      status_score = np.mean(last_scores)
      status_steps = np.mean(last_steps)
      max_last_100 = np.max(last_scores)
      #max_all = np.max(all_scores)
      all_avg_scores.append(status_score)
      all_avg_steps.append(status_steps)
      nz_rew_prc = self.memory.get_reward_sparsity()
      if (n_ep_per_update >0) and ((i_episode % n_ep_per_update) == 0):
        for upd in range(n_update_per_ep):
          self.train()
      print("\rEp {:>4}  sc:{:5.3f},  avg:{:5.3f},  nsf:{:4.2f},  steps:{:>3},  nzr:{:4.1f}%    ".format(
          i_episode, last_score, status_score,
          self.noise_scaling_factor, ts, 
          nz_rew_prc * 100), end='', flush=True)
      if (i_episode > 0) and (i_episode % 100) == 0:
        print("\rEp {:>4} sc:{:5.3f}  avg:{:5.3f}  max100:{:5.3f}  buff:{:>7}/{}  upd:{:>6}  nsf:{:4.2f}  avg_stp:{:>5.1f}  nzr:{:4.1f}%".format(
            i_episode, last_score, status_score, max_last_100,
            len(self.memory), self.memory.capacity, self.n_updates,
            self.noise_scaling_factor, status_steps,  
            nz_rew_prc * 100), flush=True)
      if (status_score >= 0.5) and (solved == 0):
        print("\nEnvironment solved in {} episodes!".format(i_episode+1))  
        self.save()
        solved = i_episode+1
    return solved, all_scores, all_avg_scores, all_avg_steps
  