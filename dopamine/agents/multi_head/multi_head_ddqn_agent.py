"""Multi Head DQN agent."""

import collections
import os

from dopamine.agents.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow as tf
import numpy as np


@gin.configurable
class MultiHeadDDQNAgent(dqn_agent.DQNAgent):
  """DQN agent with multiple heads."""

  def __init__(self,
               sess,
               num_actions,
               num_heads=1,
               transform_strategy='IDENTITY',
               num_convex_combinations=1,
               network=atari_helpers.multi_head_network,
               init_checkpoint_dir=None,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_heads: int, Number of heads per action output of the Q function.
      transform_strategy: str, Possible options include (1)
      'STOCHASTIC' for multiplication with a left stochastic matrix. (2)
      'IDENTITY', in which case the heads are not transformed.
      num_convex_combinations: If transform_strategy is 'STOCHASTIC',
        then this argument specifies the number of random
        convex combinations to be created. If None, `num_heads` convex
        combinations are created.
      network: function expecting three parameters: (num_actions, network_type,
        state). This function will return the network_type object containing the
        tensors output by the network. See atari_helpers.multi_head_network as
        an example.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      **kwargs: Arbitrary keyword arguments.
    """
    tf.logging.info('Creating MultiHeadDQNAgent with following parameters:')
    tf.logging.info('\t num_heads: %d', num_heads)
    tf.logging.info('\t transform_strategy: %s', transform_strategy)
    tf.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
    tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)

    self.num_heads = num_heads
    
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self._q_heads_transform = None
    self._num_convex_combinations = num_convex_combinations
    self.transform_strategy = transform_strategy
    super(MultiHeadDDQNAgent, self).__init__(
        sess, num_actions, network=network, **kwargs)
    
  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.
    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_actions, self.num_heads, 
                           self._get_network_type(), name=name)
    return network

  def _get_network_type(self):
    """Returns the type of the outputs of a Q value network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('multi_head_DQN_network',
                                  ['q_heads', 'unordered_q_heads', 'q_values'])

  def _network_template(self, state):
    kwargs = {}
    if self._q_heads_transform is None:
      if self.transform_strategy == 'STOCHASTIC':
        tf.logging.info('Creating q_heads transformation matrix..')
        self._q_heads_transform = atari_helpers.random_stochastic_matrix(
            self.num_heads, num_cols=self._num_convex_combinations)
    if self._q_heads_transform is not None:
      kwargs.update({'transform_matrix': self._q_heads_transform})
    return self.network(self.num_actions, self.num_heads,
                        self._get_network_type(), state,
                        self.transform_strategy, **kwargs)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # q_values is the mean of the heads
    q_values_next = self.online_convnet(self._replay.next_states).q_values
    best_actions= tf.argmax(q_values_next, axis=1)
    next_q_heads_target = self.target_convnet(self._replay.next_states).q_heads
    q_values_next_target = tf.reduce_min(next_q_heads_target, axis=-1)
    bb = tf.stack([np.arange(best_actions.get_shape().as_list()[0]),
                   best_actions], axis=-1)
    x = self._replay.rewards + self.cumulative_gamma * \
        tf.gather_nd(q_values_next_target, bb) * (
        1. - tf.cast(self._replay.terminals, tf.float32))
    x = tf.expand_dims(x, axis=1)
    return tf.tile(x, [1, self.num_heads])

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    actions = self._replay.actions
    indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
    replay_chosen_q = tf.gather_nd(
        self._replay_net_outputs.q_heads, indices=indices)
    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    q_head_losses = tf.reduce_mean(loss, axis=0)
    final_loss = tf.reduce_mean(q_head_losses)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', final_loss)
    return self.optimizer.minimize(final_loss)
