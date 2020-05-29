import tensorflow as tf
import numpy as np

class linUCB_network:

  def __init__(self, X, d, K, alpha, reward_ph):
    self.reward_ph = reward_ph
    self.A_inv = tf.Variable(tf.eye(d))
    self.b = tf.Variable(tf.zeros((d, 1)))
    self.theta = self.A_inv@self.b

    S = [alpha*tf.sqrt(tf.transpose(X[i])@self.A_inv@X[i]) 
         for i in range(K)]
    self.P = tf.squeeze(tf.stack(
                       [tf.transpose(self.theta)@X[i] + S[i]
                        for i in range(K)]))

    self.max_ucb = tf.reduce_max(self.P)
    self.best_action = tf.argmax(self.P)

    X_a = tf.gather(X, self.best_action)
    new_A_inv = self.A_inv - tf.divide(self.A_inv@X_a@tf.transpose(X_a)@self.A_inv,
                                       1 + tf.transpose(X_a)@self.A_inv@X_a)

    self.update_A_inv = tf.assign(self.A_inv, new_A_inv)
    self.update_b = tf.assign(self.b, self.b + X_a*reward_ph)

class linUCB_env:
  def __init__(self, sess, state_ph, X, d, K, alpha, num_agents, reward_ph):
    self._sess = sess
    self.state_ph = state_ph

    self.ucb_nets = [linUCB_network(X, d, K, alpha, reward_ph)
                     for _ in range(num_agents)]

    self.agents_evaluations = tf.squeeze(tf.stack([x.max_ucb for x in self.ucb_nets]))
    self.cur_agent = 0

  def update_ucb(self, reward, state):
    x = self._sess.run([self.agents_evaluations] +
                       [agent.update_A_inv for agent in self.ucb_nets] +
                       [self.ucb_nets[self.cur_agent].update_b],
                       {self.ucb_nets[self.cur_agent].reward_ph: reward,
                        self.state_ph: state}
                      )
    # break ties randomly
    self.cur_agent = np.random.choice(np.flatnonzero(x[0] == x[0].max()))
