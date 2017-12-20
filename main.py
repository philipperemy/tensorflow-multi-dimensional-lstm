import sys
from time import time

import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn import dynamic_rnn

from data_gen import next_batch
from md_lstm import *


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def multi_dimensional_lstm(input_data, rnn_size):
    rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=rnn_size, input_data=input_data, sh=[1, 1])
    return rnn_out


def standard_lstm(input_data, rnn_size):
    b, h, w, c = input_data.get_shape().as_list()
    new_input_data = tf.reshape(input_data, (b, h * w, c))
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out


def run(m_id):
    use_multi_dimensional_lstm = (m_id == 1)

    anisotropy = False
    learning_rate = 0.001 * 0.5
    batch_size = 16
    h = 32
    w = 32
    channels = 1
    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    y = tf.placeholder(tf.float32, [batch_size, h, w, channels])

    hidden_size = 4
    if use_multi_dimensional_lstm:
        print('Using Multi Dimensional LSTM !')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    else:
        print('Using Standard LSTM !')
        rnn_out = standard_lstm(input_data=x, rnn_size=hidden_size)

    model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=None)

    loss = 1e4 * tf.reduce_mean(tf.abs(tf.subtract(y, model_out)))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    
    train_writer = tf.summary.FileWriter('Tensorboard_out' + '/MDLSTM',sess.graph)
    #train_writer.add_graph(sess.graph)

    steps = 100000
    for i in range(steps):
        batch = next_batch(batch_size, h, w, anisotropy)
        st = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)

        if not use_multi_dimensional_lstm and i == 0:
            print('Shuffling the batch in the height dimension for the standard LSTM.'
                  'Its like having h LSTM on the width axis.')
            perms = np.random.permutation(list(range(w)))
            batch_x = batch_x[:, perms, :, :]
            batch_y = batch_y[:, perms, :, :]
            pass

        loss_val, _ = sess.run([loss, grad_update], feed_dict={x: batch_x, y: batch_y})
        print('steps = {0} | loss = {1:.3f} | time {2:.3f}'.format(str(i).zfill(3),
                                                                   loss_val,
                                                                   time() - st))


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please specify Model 0: LSTM, 1: MD LSTM'
    model_id = int(sys.argv[1])
    run(model_id)
