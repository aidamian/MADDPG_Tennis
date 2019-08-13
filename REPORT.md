#  MADDPG implementation in PyTorch of Collaboration and Competition

## Introduction

This implementation is based on a straight-forward Multi-Agent implementation of DDPG. We have two agents each with its own actor and critic (including the target networks for each) where each critic receives the combined state as well as combined actions.

## Implementation details

The modules are divided as follows:
 - `engine.py` contains the main Multi-Agent code including the creation of each individual agent actors and critics (online & target)
 - `agent.py` defines the basic behavior of individual agents without the actual training procedure
 - `brain.py` defines the graphs that are used for actors and critics
 - `ma_per.py` contains the replay buffer definition including the prioritized experience buffers (_work in progress_)
 - `OUNoise.py` contains the Ornstein–Uhlenbeck process noise generation class
 - `main.py` is the local version of the main code that can be used locally (without the Jupyter Notebook)
 
The overall Multi-Agent DDPG approach follows the proposed approach from _[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)_. Basically, each of the two agents optimize their online critics and then use the critic direct output to perform gradient descent in the inverse direction of the critic's output.
![MADDPG](img/maddpg.png)
The training pseudo-code can be summarized as:
```
  for each agent:
    both agents target actors computes next_action
    curernt agent target critic generates next q-value
    using current agent reward compute TD-error
    optimize current agent critic
    compute next action for all actors 
    compute q-value with current agent critic (using all actions and states)
    minimize q-value return wrt current agent actor parameters
  for each agent:
    slowly transfer weights from online to target graphs
```
### The grid-search

In order to find optimal solution we performed multi-step grid-search. The first grid-search dictionary has been defined with the following self-explanatory parameters:

```
  dct_grid = {
      "actor_layers" : [ 
              [128, 128],
            ],
      "critic_state_layers" : [
            [128],
          ],
      "critic_final_layers" : [
            [128],
          ],
      "actor_input_bn" : [True, False],
      "actor_hidden_bn" : [True, False],
      "critic_state_bn" : [True, False],
      "critic_final_bn" : [True, False],
      "apply_post_bn"   : [True, False],
      "noise_scaling_factor" : [2, 1],
      }
```
*_Important note: the purpose of the grid-search was not to find a actual solution but to find a set of hyperparmeters that assure a very good convergence. That is why for the grid search procedure we only called the training method of the multi-agent engine once after each episode._*
The initial graphs design has been simplified to a 3 hidden-layer for the actor (128-128-2) and a 128-units layer for the state featurization followed by another two layers (128-1) that receive the state features and the concatenated actions. This is a a considerable difference from the original DDPG architecture, described in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), that specifies 400-300-2 for the actor and 400/400/200-200-1 for the critic.
The grid-search resulted in 64 iterations and best results have been obtained for:
 - `actor_hidden_bn=True`  : apply batch-norm after each hidden layer in actor graphs
 - `apply_post_bn=False`   : apply batch-norm before the non-linearity
 - `noise_scaling_factor=2`: start from a scaling factor of 2 for the Ornstein–Uhlenbeck noise generation and slowly decrease up to a 0.5x noise

```
  {'actor_input_bn': True,  'actor_hidden_bn': True, 'critic_state_bn': False, 'critic_final_bn': True,  'apply_post_bn': False, 'noise_scaling_factor': 2}
  {'actor_input_bn': True, ' actor_hidden_bn': True, 'critic_state_bn': True,  'critic_final_bn': False, 'apply_post_bn': True,  'noise_scaling_factor': 2}
  {'actor_input_bn': False, 'actor_hidden_bn': True, 'critic_state_bn': True,  'critic_final_bn': False, 'apply_post_bn': False, 'noise_scaling_factor': 2}
  {'actor_input_bn': False, 'actor_hidden_bn': True, 'critic_state_bn': True,  'critic_final_bn': False, 'apply_post_bn': False, 'noise_scaling_factor': 2}
  {'actor_input_bn': False, 'actor_hidden_bn': True, 'critic_state_bn': False, 'critic_final_bn': True,  'apply_post_bn': False, 'noise_scaling_factor': 1}
  {'actor_input_bn': False, 'actor_hidden_bn': True, 'critic_state_bn': False, 'critic_final_bn': True,  'apply_post_bn': False, 'noise_scaling_factor': 2} 
```

The second stage after narrowing the first set of parameters was to introduce `selu` and `elu` non-linearities as well as allow the noise to be generated from a Gaussian rather from the OU process. 
No improvement has been made over initial `relu` activation by using the new activations however the generation of gaussian noise yielded a big improvement in exploration as well as convergence speed.

### The final training

The final training was done with the following graphs architecture and following hyperparameters: 
For the actor graph:
```
```
For the critic graph:
```
```
Following a warming-up of 4096 steps/observations the training procedure is called by the environment loop each step.
The final results for 1000 episodes is below:
![FinalResults][img/MADDPG_1.png]
as well as the training history where we can observe for each 100 episodes the final score, the 100-running average score, the 100-running max, overall max score, the loaded buffer size, number of training updates, the noise scaling factors `nsf` and the 100-running mean steps per episode
```
```


## Future improvements

Work is already underway for a implementation of a prioritized experience buffer that will be based/updated using the TD-error generated by the individual critics (we sample separate batches of experiences for each individual agent).
A second improvement will be the addition of second critic for each agent in order to implement the TD3 approach.