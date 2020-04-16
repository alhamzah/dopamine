"""Multi Head UCB DQN agent."""

import collections
import os
import math

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

  def _compute_q_argmax(self):
    self.cur_head = np.argmax(self._compute_ucb())
    x = self._sess.run(self._net_outputs.q_heads,
                       {self.state_ph: self.state})
    return np.argmax(x[:,:,self.cur_head], axis=1)[0]
    
  def _update_ucb(self, selected_head, reward):
    self.ucb_count += 1
    self.arms[selected_head]['count'] += 1
    reward = int(reward > self.ucb_cutoff)
    self.arms[selected_head]['sum'] += reward
    self.rewards[selected_head].pop()
    self.rewards[selected_head].appendleft(reward)
    
  def _store_transition(self, last_observation, action, reward, is_terminal):
    self._update_ucb(self.cur_head, reward)
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
        with open(self.log_dir+'/ucb_values.txt', 'ab') as f:
          x = self._compute_ucb()
          np.savetxt(f, np.array([x]))

      self.log_counter += 1
