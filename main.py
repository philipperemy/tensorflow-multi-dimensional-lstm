from time import time

import numpy as np

from md_lstm import *


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def run():
    learning_rate = 0.001
    batch_size = 16
    h = 32
    w = 32
    channels = 4
    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    y = tf.placeholder(tf.float32, [batch_size, 10])

    def next_batch():
        return np.random.uniform(size=(batch_size, h, w, channels)), np.random.randint(low=0, high=2,
                                                                                       size=(batch_size, 10))

    hidden_size = 64
    outputs, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[2, 2])
    rnn_out = tf.squeeze(outputs[:, -1, -1, :])

    fc0_w = create_weight_variable('fc0_w', [hidden_size, 10])
    fc0_b = create_bias_variable('fc0_b', [10])
    y = tf.matmul(rnn_out, fc0_w) + fc0_b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    steps = 100
    for i in range(steps):
        batch = next_batch()
        st = time()
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update],
                                      feed_dict={x: batch[0], y: batch[1]})
        print('Forward-Backward pass took {0:.2f}s to complete.'.format(time() - st))


if __name__ == '__main__':
    run()
