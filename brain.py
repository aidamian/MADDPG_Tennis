# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn



def hidden_init(layer):
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)

def init_layers(layers, init_custom):
  for layer in layers:
    if hasattr(layer,"weight"):
      if init_custom:
          layer.weight.data.uniform_(*hidden_init(layer))
      else:
          nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer,"bias"):
      if layer.bias is not None:
          if init_custom:
              layer.bias.data.uniform_(*hidden_init(layer))
          else:
              nn.init.xavier_uniform_(layer.bias)


class MADDPGActor(nn.Module):
  def __init__(self, input_size, output_size, 
               layers=[256, 128], # 256 128
               init_custom=True, use_input_bn=False, use_hidden_bn=False,
               bn_post=False,
               model_name='actor',
               activation='relu'):
    """
    The configurable Actor module
    layers: configurable stream of layers - allways outputs a tanh
    init_custom : will use uniform distrib  1 / sqrt(fan_in) instead of xavier uniform initialization
    """
    super(MADDPGActor, self).__init__()
    self.layers = nn.ModuleList()
    self.init_custom = init_custom
    self.model_name = model_name
    pre_units = input_size
    if use_input_bn:
      self.layers.append(nn.BatchNorm1d(pre_units))
    
    for i,L in enumerate(layers):
      if i == 0:
          use_bias = not use_input_bn
      else:
          use_bias = True
      self.layers.append(nn.Linear(pre_units, L, bias=use_bias))
      if use_hidden_bn and not bn_post:
          self.layers.append(nn.BatchNorm1d(L))
          
      if activation == 'relu':
        self.layers.append(nn.ReLU())
      elif activation == 'elu':
        self.layers.append(nn.ELU())
      elif activation == 'selu':
        self.layers.append(nn.SELU())
      else:
        raise ValueError("Unknown activation!")
        
      if use_hidden_bn and bn_post:
          self.layers.append(nn.BatchNorm1d(L))
          
      pre_units = L
    self.final_linear = nn.Linear(pre_units, output_size)
    self.final_activation = nn.Tanh()
    self.reset_parameters()
    return
      
      
  def reset_parameters(self):            
    init_layers(self.layers, self.init_custom)                    
    nn.init.uniform_(self.final_linear.weight, -0.003, 0.003)  
    nn.init.uniform_(self.final_linear.bias, -0.003, 0.003)  
    return
  
      
  def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    x = self.final_linear(x)
    x = self.final_activation(x)
    return x
  

class MADDPGCritic(nn.Module):
  def __init__(self, state_size, act_size, output_size=1, leaky=False, init_custom=True,
               act_layers=[], 
               state_layers=[256],  # 256
               final_layers=[256, 128], # 256 128
               state_bn=True,
               bn_post=False,
               act_bn=False,
               other_bn=False,
               model_name='critic',
               activation='relu',
               ):
    """
    The Critic module can be easily configured to almost any number of layers. It currently support three main
    graph streams: 
    act_layers   : the action stream that is by default with no layers. This streams processes the action input
    state_layers : the state stream is responsible for the generation of high-level features for the observation
    final_layers : the final stream that gets the concatenated action/state streams and generates the value
    leay : will use LeakyReLU for all activations instead of ReLU
    init_custom : will use uniform distrib  1 / sqrt(fan_in) instead of xavier uniform initialization
    """
    super(MADDPGCritic, self).__init__()
    self.init_custom = init_custom
    self.model_name = model_name
    
    self.final_layers = nn.ModuleList()

    pre_units = state_size
    if len(state_layers) > 0:
        self.state_layers = nn.ModuleList()
        for L in state_layers:
            self.state_layers.append(nn.Linear(pre_units, L))
            if state_bn and not bn_post:
              self.state_layers.append(nn.BatchNorm1d(L))
            if leaky:
              self.state_layers.append(nn.LeakyReLU())
            else:
              if activation == 'relu':
                self.state_layers.append(nn.ReLU())
              elif activation == 'elu':
                self.state_layers.append(nn.ELU())
              elif activation == 'selu':
                self.state_layers.append(nn.SELU())
              else:
                raise ValueError("Unknown activation!")
                
            if state_bn and bn_post:
                self.state_layers.append(nn.BatchNorm1d(L))
            pre_units = L
    final_state_column_size = pre_units

    pre_units = act_size
    if len(act_layers) > 0:
        self.act_layers = nn.ModuleList()
        for L in act_layers:
            self.act_layers.append(nn.Linear(pre_units, L))
            if act_bn and not bn_post:
              self.act_layers.append(nn.BatchNorm1d(L))                    
            if leaky:
              self.act_layers.append(nn.LeakyReLU())
            else:
              if activation == 'relu':
                self.act_layers.append(nn.ReLU())
              elif activation == 'elu':
                self.act_layers.append(nn.ELU())
              elif activation == 'selu':
                self.act_layers.append(nn.SELU())
              else:
                raise ValueError("Unknown activation!")

            if act_bn and bn_post:
                self.act_layers.append(nn.BatchNorm1d(L))                    
            pre_units = L
    final_action_column_size = pre_units
        
    pre_units = final_state_column_size + final_action_column_size
    for L in final_layers:
        self.final_layers.append(nn.Linear(pre_units, L))
        if other_bn and not bn_post:
          self.final_layers.append(nn.BatchNorm1d(L))                    
        if leaky:
          self.final_layers.append(nn.LeakyReLU())
        else:
          if activation == 'relu':
            self.final_layers.append(nn.ReLU())
          elif activation == 'elu':
            self.final_layers.append(nn.ELU())
          elif activation == 'selu':
            self.final_layers.append(nn.SELU())
          else:
            raise ValueError("Unknown activation!")

        if other_bn and bn_post:
            self.final_layers.append(nn.BatchNorm1d(L))                    
        pre_units = L        
        
    self.final_linear = nn.Linear(pre_units, output_size)
    self.reset_parameters()
    return
  
      
  def reset_parameters(self):
    if hasattr(self, "state_layers"):
        init_layers(self.state_layers, self.init_custom)
        
    if hasattr(self, "act_layers"):
        init_layers(self.act_layers, self.init_custom)
        
    init_layers(self.final_layers, self.init_custom)
        
    nn.init.uniform_(self.final_linear.weight, -0.003, 0.003)
    nn.init.uniform_(self.final_linear.bias, -0.003, 0.003)
    return
      
      
  def forward(self, state, action):
    x_state = state
    if hasattr(self, "state_layers"):
        for layer in self.state_layers:
            x_state = layer(x_state)
        
    x_act = action
    if hasattr(self, "act_layers"):
        for layer in self.act_layers:
            x_act = layer(x_act)
    
    x = th.cat((x_state, x_act), dim=1)
    
    for layer in self.final_layers:
        x = layer(x)    
    
    x = self.final_linear(x)
    return x


    

def layers_stats(model):
  VANISH_THR = 1e-9
  VANISH_NR = 3
  vanished = 0
  msg = ""
  for name, param in model.named_parameters():
      data = param.detach().cpu().numpy()
      grads = param.grad.detach().cpu().numpy()
      status_grad = ""
      if (grads.max() > -VANISH_THR) and (grads.max() < VANISH_THR):
        status_grad = 'VANISH'
      if (grads.min() < -1) or (grads.min() > 1):
        status_grad = 'EXPLODE'
      if status_grad != '':
        vanished += 1
        msg += "  {:<25} {:<7} {:>8.1e} / {:>8.1e} / {:>8.1e} / {:>8.1e} {}...\n".format(
            name+":", "", data.min(), data.max(), data.mean(), np.median(data), data.ravel()[:2])
        msg += "  {:<25} {:<7} {:>8.1e} / {:>8.1e} / {:>8.1e} / {:>8.1e} {}...\n".format(
            "    grads:", status_grad, grads.min(), grads.max(), grads.mean(), np.median(grads), grads.ravel()[:2])
      if vanished >= VANISH_NR:
        print("")
        print("Model {} min/max/mean/median:".format(model.model_name))
        print(msg)
        return True
  return False
      
      