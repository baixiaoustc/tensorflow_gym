__author__ = 'baixiao'
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    Q_value = tf.placeholder(tf.float32, [None, 6])
    action_input = tf.placeholder(tf.float32, [None, 6])
    Q_action = tf.reduce_sum(tf.multiply(Q_value, action_input), reduction_indices=1)
    y_input = tf.placeholder(tf.float32, [None])
    cost = tf.reduce_mean(tf.square(y_input - Q_action))
    # optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Q_batch = [[1,0,0,0,0,0]]
        action_batch = [[0,1,0,0,0,0]]
        y_batch = [1.1]
        qv, ai, qa, yi, cost = sess.run([Q_value, action_input, Q_action,y_input, cost], {Q_value: Q_batch, action_input: action_batch, y_input: y_batch})
        print qv
        print(ai)
        print qa
        print yi
        print cost