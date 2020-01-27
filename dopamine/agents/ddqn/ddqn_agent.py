# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of a DDQN agent."""

from dopamine.agents.dqn import dqn_agent
import numpy as np
import tensorflow.compat.v1 as tf

import gin.tf

@gin.configurable
class DDQNAgent(dqn_agent.DQNAgent):
  def __init__(self, sess, num_actions, summary_writer=None):
    super(DDQNAgent, self).__init__(sess=sess,
                                    num_actions=num_actions,
                                    summary_writer=summary_writer)
    
  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    q_values_next = self.online_convnet(self._replay.next_states)
    best_actions = tf.math.argmax(tf.squeeze(q_values_next), axis=1)
    q_values_next_target = self.target_convnet(self._replay.next_states)
    bb = tf.stack([np.arange(best_actions.get_shape().as_list()[0]), tf.squeeze(best_actions)], axis=-1)
    return self._replay.rewards + self.cumulative_gamma * \
        tf.gather_nd(tf.squeeze(q_values_next_target), bb) * (
        1. - tf.cast(self._replay.terminals, tf.float32))
