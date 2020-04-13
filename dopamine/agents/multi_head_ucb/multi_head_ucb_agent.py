"""Multi Head UCB DQN agent."""

import collections
import os
import math

from dopamine.agents.multi_head import atari_helpers
from dopamine.agents.multi_head import multi_head_ddqn_agent
import gin
import tensorflow as tf
import numpy as np

import random

@gin.configurable
class MultiHeadUCBAgent(multi_head_ddqn_agent.MultiHeadDDQNAgent):
  """DQN agent with multiple heads and UCB to select the best head."""

  def __init__(self,
               sess,
               num_actions,
               ucb_cutoff=0.5,
               ucb_horizon=100,
               ucb_lambda=0.9,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.
    """
    tf.logging.info('Creating MultiHeadUCBAgent with following parameters:')
    tf.logging.info('\t ucb_cutoff: %f', ucb_cutoff)
    tf.logging.info('\t ucb_horizon: %d', ucb_horizon)
    tf.logging.info('\t ucb_lambda: %f', ucb_lambda)
    
    self.ucb_cutoff = ucb_cutoff
    self.ucb_horizon = ucb_horizon
    self.ucb_lambda = ucb_lambda
    self.cur_head = 0
    
    super(MultiHeadUCBAgent, self).__init__(
        sess, num_actions, **kwargs)
    
    self.ucb_count = 0
    self.arms = [{'sum': 0, 'count': 0} for _ in range(self.num_heads)]
    self.rewards = [collections.deque([0]*self.ucb_horizon)
                    for _ in range(self.num_heads)]
    self._update_q_argmax()

  def _compute_ucb(self):
    bounds = []
    for arm in range(self.num_heads):
        if self.arms[arm]['count'] == 0:
            bounds.append(float('inf'))
            continue
        avg = sum([(self.ucb_lambda**i)*self.rewards[arm][i]
                    for i in range(self.ucb_horizon)])
        ucb = math.sqrt(2*math.log(self.ucb_count)/self.arms[arm]['count'])
        bounds.append(avg + ucb)
    return bounds

  def _update_q_argmax(self):
    self.cur_head = np.argmax(self._compute_ucb())
    x = tf.gather(self._net_outputs.q_heads, self.cur_head, axis=2)
    self._q_argmax = tf.argmax(x, axis=1)[0]
    
  def _update_ucb(self, selected_head, reward):
    self.ucb_count += 1
    self.arms[selected_head]['count'] += 1
    reward = int(reward > self.ucb_cutoff)
    self.arms[selected_head]['sum'] += reward
    self.rewards[selected_head].pop()
    self.rewards[selected_head].appendleft(reward)
    self._update_q_argmax()
    
  def _store_transition(self, last_observation, action, reward, is_terminal):
    self._update_ucb(self.cur_head, reward)
    self._replay.add(last_observation, action, reward, is_terminal)
