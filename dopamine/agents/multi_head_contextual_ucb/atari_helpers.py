import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

class multi_head_contextual_ucb_network(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, num_heads, network_type, name=None, **kwargs):
    """Creates the layers used for calculating Q-values.
    """
    super(multi_head_contextual_ucb_network, self).__init__(name=name)

    self.num_actions = num_actions
    self.network_type = network_type
    self.num_heads = num_heads
    
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions*num_heads, 
      activation=None,
      name='fully_connected_q_heads')

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    ucb_context = x
    x = self.dense2(x)

    q_heads = tf.reshape(x, [-1, self.num_actions, self.num_heads])
    unordered_q_heads = q_heads
    q_values = tf.reduce_mean(q_heads, axis=-1)

    return self.network_type(q_heads, unordered_q_heads, q_values, ucb_context)

class contextual_ucb_network:

    def __init__(self, ucb_A, ucb_b, ucb_d, ucb_alpha, ucb_X):
        self.X = tf.reshape(ucb_X, (512, 1))
        self.A_inv = tf.linalg.inv(ucb_A)
        self.s = ucb_alpha*tf.sqrt(tf.transpose(self.X)@self.A_inv@self.X)
        self.theta = self.A_inv@ucb_b
#         self.P = tf.transpose(self.theta)@self.X + s
        self.P = tf.transpose(self.theta)@self.X
        self._P_argmax = tf.argmax(self.P)
