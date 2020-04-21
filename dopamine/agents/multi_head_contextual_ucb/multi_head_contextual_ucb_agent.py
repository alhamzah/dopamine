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
               ucb_alpha=3.5,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.
    """
    tf.logging.info('Creating MultiHeadUCBAgent with following parameters:')
    tf.logging.info('\t ucb_cutoff: %f', ucb_cutoff)
    tf.logging.info('\t ucb_alpha: %f', ucb_alpha)

    self.ucb_cutoff = ucb_cutoff
    self.ucb_d = 512
    self.ucb_alpha = np.sqrt(1/2*np.log(2*(2*10**6 *
                                          8/0.01)))

    super(MultiHeadContextualUCBAgent, self).__init__(
        sess, num_actions, network=network, **kwargs)

    self.cur_head = 0
    self.ucb_count = 0
    self.ucb_A = np.identity(self.ucb_d)
    self.ucb_b = np.zeros((self.ucb_d, self.num_heads))

  def _build_networks(self):
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)

    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    self.ucb_A_ph = tf.placeholder(tf.float32)
    self.ucb_b_ph = tf.placeholder(tf.float32)
    self.ucb_net = atari_helpers.contextual_ucb_network(self.ucb_A_ph,
                                                        self.ucb_b_ph,
                                                        self.ucb_d,
                                                        self.ucb_alpha,
                                                        self._net_outputs.ucb_context)

  def _get_network_type(self):
    return collections.namedtuple('multi_head_contextual_ucb_network',
                                  ['q_heads', 'unordered_q_heads', 'q_values',
                                   'ucb_context'])

  def _compute_q_argmax(self):
    """Update the best head and find the best action in this context"""
    self.cur_head = self._sess.run(self.ucb_net._P_argmax,
                                   {self.state_ph: self.state,
                                    self.ucb_A_ph: self.ucb_A,
                                    self.ucb_b_ph: self.ucb_b})[0]
    x = self._sess.run(self._net_outputs.q_heads,
                       {self.state_ph: self.state})
    return np.argmax(x[:,:,self.cur_head], axis=1)[0]

  def _update_ucb(self, selected_head, reward):
    import pdb; pdb.set_trace()
    r = int(reward > self.ucb_cutoff)
    ucb_X = self._sess.run(self.ucb_net.X,
                           {self.state_ph: self.state})
    self.ucb_A += ucb_X.dot(ucb_X.T)
    self.ucb_b[:, selected_head] += np.squeeze(ucb_X*r)

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
          x = self._sess.run(self.ucb_net.P,
                             {self.state_ph: self.state,
                              self.ucb_A_ph: self.ucb_A,
                              self.ucb_b_ph: self.ucb_b})
          np.savetxt(f, x.T)

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
    return bundle_dictionary
