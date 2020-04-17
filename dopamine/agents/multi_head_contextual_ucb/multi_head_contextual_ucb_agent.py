"""Multi Head UCB DQN agent."""

import collections
import os
import math

from dopamine.agents.multi_head import multi_head_ddqn_agent
from dopamine.agents.multi_head_contextual_ucb import atari_helpers 
import gin
import tensorflow as tf
import numpy as np

import random

@gin.configurable
class MultiHeadContextualUCBAgent(multi_head_ddqn_agent.MultiHeadDDQNAgent):
  """DQN agent with multiple heads and LinUCB to select the best head."""

  def __init__(self,
               sess,
               num_actions,
               network=atari_helpers.multi_head_contextual_ucb_network,
               ucb_cutoff=0.5,
               ucb_alpha=0.9,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.
    """
    tf.logging.info('Creating MultiHeadUCBAgent with following parameters:')
    tf.logging.info('\t ucb_cutoff: %f', ucb_cutoff)
    tf.logging.info('\t ucb_alpha: %f', ucb_alpha)

    self.ucb_cutoff = ucb_cutoff
    self.ucb_alpha = ucb_alpha
    self.cur_head = 0

    super(MultiHeadContextualUCBAgent, self).__init__(
        sess, num_actions, network=network, **kwargs)

    self.ucb_count = 0
    self.ucb_d = 512
    self.ucb_A = np.identity(self.ucb_d)
    self.ucb_b = np.zeros(self.ucb_d).reshape((self.ucb_d, 1))
    self.ucb_X = np.zeros(self.ucb_d).reshape((self.ucb_d, 1))

  def _get_network_type(self):
    return collections.namedtuple('multi_head_contextual_ucb_network',
                                  ['q_heads', 'unordered_q_heads', 'q_values',
                                   'ucb_context'])

  def _compute_ucb(self):
    A_inv = np.linalg.inv(self.ucb_A)
    theta = np.dot(A_inv, self.ucb_b).reshape((self.ucb_d, 1))
    self.ucb_X = self._sess.run(self._net_outputs.ucb_context,
                                {self.state_ph: self.state})
    self.ucb_X = self.ucb_X.reshape((self.ucb_d, 1))
    X = self.ucb_X

    bounds = []
    for arm in range(self.num_heads):
        p_a = np.dot(theta.T, X) + self.ucb_alpha*np.sqrt(X.T.dot(A_inv).dot(X))
        bounds.append(p_a[0,0])
    return bounds

  def _compute_q_argmax(self):
    self.cur_head = np.argmax(self._compute_ucb())
    x = self._sess.run(self._net_outputs.q_heads,
                       {self.state_ph: self.state})
    return np.argmax(x[:,:,self.cur_head], axis=1)[0]
    
  def _update_ucb(self, selected_head, reward):
    r = int(reward > self.ucb_cutoff)
    self.ucb_A += self.ucb_X.dot(self.ucb_X.T)
    self.ucb_b += self.ucb_X*r
    
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

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['training_steps'] = self.training_steps
    bundle_dictionary['ucb_count'] = self.ucb_count
    bundle_dictionary['ucb_A'] = self.ucb_A
    bundle_dictionary['ucb_b'] = self.ucb_b
    bundle_dictionary['ucb_X'] = self.ucb_X
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      tf.logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
          tf.logging.info('\t loaded ' + key)
    elif not self.allow_partial_reload:
      return False
    else:
      tf.logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
