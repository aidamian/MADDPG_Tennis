# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 07:54:58 2019

@author: Andrei
"""


from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from time import time

import torch as th

from engine import MADDPGEngine

from ma_per import MASimpleReplayBuffer

def play_env(env, brain_name, train=False, ma_eng=None, num_episodes=5, chk_mem=False):
  if train:
    s_pmode = "non-visualy"
  else:
    s_pmode = "visualy"
    
  if ma_eng is not None:
    s_amode = 'agent-based'
  else:
    s_amode = 'randomly'
    
  print("\n\nPlaying {} the enviroment for {} eps in {} mode...".format(
      s_pmode, num_episodes, s_amode))

  t_start = time()
  
  if chk_mem:
    temp_mem = MASimpleReplayBuffer(capacity=10000, nr_agents=num_agents,
                                    device=th.device('cuda:0'))

  for i in range(1, 6):                                    # play game for 5 episodes
    env_info = env.reset(train_mode=train)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
      if ma_eng is not None:
        actions = ma_eng.act(states, add_noise=False)
      else:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
      env_info = env.step(actions)[brain_name]           # send all actions to tne environment
      next_states = env_info.vector_observations         # get next state (for each agent)
      rewards = env_info.rewards                         # get reward (for each agent)
      dones = env_info.local_done                        # see if episode finished
      if chk_mem:
        temp_mem.add(states, actions, rewards, next_states, dones)
      scores += rewards                                  # update the score (for each agent)
      states = next_states                               # roll over states to next time step
      if np.any(dones):                                  # exit loop if episode finished
        break
    print('  Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

  t_end = time()
  print("Done playing {} {} the enviroment for {} eps in {:.3f}s".format(
      s_amode, s_pmode, num_episodes, t_end-t_start))
  
  if chk_mem:
    sample = temp_mem.sample(10)
    print("Sampled buffer: {}".format(len(sample)))
    for i,x in enumerate(sample[0]):
      print(" Sample[0][{}]: {}".format(i, x.shape))
  return
  

def ma_ddpg(env, ma_eng: MADDPGEngine, brain_name, n_episodes=2000,
            n_ep_per_update=2):  
  last_scores = deque(maxlen=100)
  all_scores = []
  for i_episode in range(1,n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    ep_rewards = []
    while True:
      actions = ma_eng.act(states, add_noise=True)
      env_info = env.step(actions)[brain_name]           # send all actions to tne environment
      next_states = env_info.vector_observations         # get next state (for each agent)
      rewards = env_info.rewards                         # get reward (for each agent)
      dones = env_info.local_done                        # see if episode finished
      ma_eng.add_exp(states, actions, rewards, next_states, dones)
      ep_rewards.append(rewards)
      states = next_states                               # roll over states to next time step
      if np.any(dones):                                  # exit loop if episode finished
        break
    np_rewards = np.array(ep_rewards)
    scores = np_rewards.sum(axis=0)
    last_score = scores.max()
    all_scores.append(last_score)
    last_scores.append(last_score)
    status_score = np.mean(last_scores)
    max_last_100 = np.max(last_scores)
    max_all = np.max(all_scores)
    if (i_episode % n_ep_per_update) == 0:
      ma_eng.train()
    print("\rEpisode {:>4} score: {:5.2f},  avg:{:5.1f},  nsf:{:4.2f}".format(
        i_episode, last_score, status_score, ma_eng.noise_scaling_factor), end='', flush=True)
    if (i_episode > 0) and (i_episode % 100) == 0:
      print("\rEpisode {:>4} score:{:5.2f}  avg:{:5.2f}  max100:{:5.2f}  max:{:5.2f}  buff:{:>7}/{}  upd:{}".format(
          i_episode, last_score, status_score, max_last_100, max_all,
          len(ma_eng.memory), ma_eng.memory.capacity, ma_eng.n_updates), flush=True)
    if status_score >= 0.5:
      print("\nEnvironment solved in {} episodes!".format(i_episode+1))      
  return all_scores
  


if __name__ == '__main__':
  
  """
  TODO:
    Grad clip
    lr mod
    smaller buffer
    
  """
  
  env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
  # get the default brain
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]

  # reset the environment
  env_info = env.reset(train_mode=False)[brain_name]
  
  # number of agents 
  num_agents = len(env_info.agents)
  print('  Number of agents:', num_agents)
  
  # size of each action
  action_size = brain.vector_action_space_size
  print('  Size of each action:', action_size)
  
  # examine the state space 
  states = env_info.vector_observations
  state_size = states.shape[1]
  print('  There are {} agents. Each observes a individual state with length: {}'.format(states.shape[0], state_size))
  print('  The state for the first agent looks like:', states[0])  

  
  if False:
    play_env(env)
    play_env(env, brain_name, train=True, chk_mem=True)
  
  eng = MADDPGEngine(action_size=action_size, 
                     state_size=state_size, 
                     n_agents=num_agents)
  
  ma_ddpg(env, eng, brain_name)
  
  env.close()


