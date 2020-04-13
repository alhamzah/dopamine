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
    self._update_q_argmax()

  def _thompson_sample(self):
    probs = []
    for arm in range(self.num_heads):
        p = np.random.beta(self.arms[arm]['success']+1, self.arms[arm]['failure']+1)
        probs.append(p)
    return probs

  def _update_q_argmax(self):
    self.cur_head = np.argmax(self._thompson_sample())
    x = tf.gather(self._net_outputs.q_heads, self.cur_head, axis=2)
    self._q_argmax = tf.argmax(x, axis=1)[0]
    
  def _update_thompson(self, selected_head, reward):
    reward = int(reward > self.thompson_cutoff)
    self.arms[selected_head]['success'] += reward
    self.arms[selected_head]['failure'] += 1-reward
    self._update_q_argmax()
    
  def _store_transition(self, last_observation, action, reward, is_terminal):
    self._update_thompson(self.cur_head, reward)
    self._replay.add(last_observation, action, reward, is_terminal)
