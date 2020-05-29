from linUCB import linUCB_network, linUCB_env
import tensorflow as tf
import unittest

class TestUcb(unittest.TestCase):

    def test_bad_action(self):
        alpha = 3
        d = 2
        K = 2
        contexts = tf.expand_dims(tf.eye(2), -1)

        reward_ph = tf.placeholder(tf.float32)
        agent = linUCB_network(contexts, d, K, alpha, reward_ph)
        action_count = [0,0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(1000):
                best_action = sess.run(agent.best_action)

                reward = 0
                if best_action == 1:
                    reward = 1

                sess.run([agent.update_b,
                          agent.update_A_inv],
                         {reward_ph: reward})
                action_count[best_action] += 1

            self.assertEquals(sess.run(agent.best_action), 1)

class TestUcbEnv(unittest.TestCase):

    def test_bad_agent(self):
        alpha = 3
        d = 2
        K = 2
        contexts = tf.expand_dims(tf.eye(2), -1)        
        state_ph = tf.placeholder(tf.float32)
        reward_ph = tf.placeholder(tf.float32)
        num_agents = 2

        agent_count = [0,0]
        action_count = [0,0]

        with tf.Session() as sess:
            ucb_env = linUCB_env(sess, state_ph, contexts, d, K, alpha, num_agents, reward_ph)
            sess.run(tf.global_variables_initializer())

            for i in range(1000):
                best_agent = ucb_env.cur_agent
                action = sess.run(ucb_env.ucb_nets[best_agent].best_action)
                action_count[action] += 1

                reward = 0
                if best_agent == 1:
                    reward = 1

                ucb_env.update_ucb(reward, None)
                agent_count[best_agent] += 1
            
            self.assertEquals(ucb_env.cur_agent,1)

if __name__ == '__main__':
    unittest.main()