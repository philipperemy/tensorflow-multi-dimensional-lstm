from time import time

import argparse
import logging
import numpy as np
import tensorflow.contrib.slim as slim

from data_gen_corr_fields import next_batch, visualise_mat, get_relevant_prediction_index
from md_lstm import *
from md_lstm import standard_lstm

logger = logging.getLogger(__name__)


def get_arguments(parser: argparse.ArgumentParser):
    args = None
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)
    return args


def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type', required=True, help='Valid model types are [lstm, md_lstm].')
    args = get_arguments(parser)
    logger.info(f'Script inputs: {args}.')
    return args


def run(model_type='md_lstm'):
    use_multi_dimensional_lstm = (model_type == 'md_lstm')

    learning_rate = 0.01
    batch_size = 16
    h = 8
    w = 8
    channels = 1
    hidden_size = 16

    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    y = tf.placeholder(tf.float32, [batch_size, h, w, channels])

    if use_multi_dimensional_lstm:
        logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    else:
        logger.info('Using Standard LSTM.')
        rnn_out = standard_lstm(input_data=x, rnn_size=hidden_size)

    model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.square(y - model_out))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    steps = 100000
    for i in range(steps):
        batch = next_batch(batch_size, h, w)
        st = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)

        mo, loss_val, _ = sess.run([model_out, loss, grad_update], feed_dict={x: batch_x, y: batch_y})

        """
        ____________
        |          |
        |          |
        |     x    |
        |      x <----- extract this prediction. Relevant loss is only computed for this value.
        |__________|    we don't care about the rest (even though the model is trained on all values
                        for simplicity). A standard LSTM should have a very high value for relevant loss
                        whereas a MD LSTM (which can see all the TOP LEFT corner) should perform well. 
        """

        # extract the predictions for the second x
        relevant_pred_index = get_relevant_prediction_index(batch_y)
        true_rel = np.array([batch_y[i, x, y, 0] for (i, (y, x)) in enumerate(relevant_pred_index)])
        pred_rel = np.array([mo[i, x, y, 0] for (i, (y, x)) in enumerate(relevant_pred_index)])
        relevant_loss = np.mean(np.square(true_rel - pred_rel))

        format_str = 'steps = {0} | overall loss = {1:.3f} | time {2:.3f} | relevant loss = {3:.3f}'
        logger.info(format_str.format(str(i).zfill(3), loss_val, time() - st, relevant_loss))

        if i % 500 == 0:
            visualise_mat(sess.run(model_out, feed_dict={x: batch_x})[0].squeeze())
            visualise_mat(batch_y[0].squeeze())


def main():
    args = get_script_arguments()
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    run(args.model_type)


if __name__ == '__main__':
    main()
