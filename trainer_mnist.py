from time import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from md_lstm import multi_dimensional_rnn_while_loop


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def run_lstm_mnist(hidden_size=32, batch_size=256, steps=1000):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.001
    x_ = tf.placeholder('float32', [batch_size, 784, 1])
    y_ = tf.placeholder('float32', [batch_size, 10])

    x = tf.reshape(x_, shape=(batch_size, 28, 28, 1))
    outputs, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    rnn_out = tf.squeeze(outputs[:, -1, -1, :])  # could take everything.

    fc0_w = create_weight_variable('fc0_w', [hidden_size, 10])
    fc0_b = create_bias_variable('fc0_b', [10])
    y = tf.matmul(rnn_out, fc0_w) + fc0_b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    def transform_x(_x_):
        t_x = np.expand_dims(_x_, axis=2)
        return t_x

    for i in range(steps):
        batch = mnist.train.next_batch(batch_size)
        st = time()
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update],
                                      feed_dict={x_: transform_x(batch[0]) + 1e-6, y_: batch[1]})
        print('Forward-Backward pass took {0:.2f}s to complete.'.format(time() - st))
        print(i, tr_loss, tr_acc)


def main():
    run_lstm_mnist(hidden_size=32, batch_size=16, steps=10000)


if __name__ == '__main__':
    main()
