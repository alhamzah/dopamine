"""Multi Head Thompson DQN agent."""

import collections
import os
import math

from dopamine.agents.multi_head import multi_head_ddqn_agent
import gin
import tensorflow as tf
import numpy as np

import random

@gin.configurable
class MultiHeadThompsonAgent(multi_head_ddqn_agent.MultiHeadDDQNAgent):
  """DQN agent with multiple heads and Thompson Sampling to select the best head."""

  def __init__(self,
               sess,
               num_actions,
               thompson_cutoff=0.5,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.
    """
    tf.logging.info('Creating MultiHeadThompsonAgent with following parameters:')
    tf.logging.info('\t thompson_cutoff: %f', thompson_cutoff)
    
    self.thompson_cutoff = thompson_cutoff
    
    super(MultiHeadThompsonAgent, self).__init__(
        sess, num_actions, **kwargs)

    self.arms = [{'success': 0, 'failure': 0} for _ in range(self.num_heads)]
    self.cur_head = 0

  def _thompson_sample(self):
    probs = []
    for arm in range(self.num_heads):
        p = np.random.beta(self.arms[arm]['success']+1, self.arms[arm]['failure']+1)
        probs.append(p)
    return probs

  def _compute_q_argmax(self):
    self.cur_head = np.argmax(self._thompson_sample())
    x = self._sess.run(self._net_outputs.q_heads,
                       {self.state_ph: self.state})
    return np.argmax(x[:,:,self.cur_head], axis=1)[0]
    
  def _update_thompson(self, selected_head, reward):
    reward = int(reward > self.thompson_cutoff)
    self.arms[selected_head]['success'] += reward
    self.arms[selected_head]['failure'] += 1-reward
    
  def _store_transition(self, last_observation, action, reward, is_terminal):
    self._update_thompson(self.cur_head, reward)
    self._replay.add(last_observation, action, reward, is_terminal)

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      self._log_values()
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state according
      # to the current head.
      return self._compute_q_argmax()

  def _log_values(self):
      if self.log_counter%self.log_frequency == 0:
        #save the q values
        with open(self.log_dir+'/q_heads_values.txt', 'ab') as f:
          x = self._sess.run(self._net_outputs.q_heads,
                             {self.state_ph: self.state})
          for i in range(len(x)):
            np.savetxt(f, x[i])
          np.savetxt(f, [0])
        with open(self.log_dir+'/thompson_probs.txt', 'ab') as f:
          x = self._thompson_sample()
          np.savetxt(f, np.array([x]))

      self.log_counter += 1
