import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift


class MultiDimensionalLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f1 = ln(f1, scope='f1/')
            f2 = ln(f2, scope='f2/')
            o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope

    returns [batch,h/sh[0],w/sh[1],channels*sh[0]*sh[1]] the output of the lstm
    """
    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):
        cell = MultiDimensionalLSTMCell(rnn_size)

        shape = input_data.get_shape().as_list()

        if shape[1] % sh[0] != 0:
            offset = tf.zeros([shape[0], sh[0] - (shape[1] % sh[0]), shape[2], shape[3]])
            input_data = tf.concat(1, [input_data, offset])
            shape = input_data.get_shape().as_list()
        if shape[2] % sh[1] != 0:
            offset = tf.zeros([shape[0], shape[1], sh[1] - (shape[2] % sh[1]), shape[3]])
            input_data = tf.concat(2, [input_data, offset])
            shape = input_data.get_shape().as_list()

        h, w = int(shape[1] / sh[0]), int(shape[2] / sh[1])
        features = sh[1] * sh[0] * shape[3]
        batch_size = shape[0]

        x = tf.reshape(input_data, [batch_size, h, w, features])
        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)
        x = tf.transpose(x, [1, 2, 0, 3])
        x = tf.reshape(x, [-1, features])
        x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32) * shape[0]
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

        states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size, rnn_size], tf.float32),
                                                          tf.zeros([batch_size, rnn_size], tf.float32)))

        def getindex1(t, w):
            return tf.cond(tf.less_equal(tf.constant(w), t),
                           lambda: t - tf.constant(w),
                           lambda: tf.constant(h * w))

        def getindex2(t, w):
            return tf.cond(tf.less(tf.constant(0), tf.mod(t, tf.constant(w))),
                           lambda: t - tf.constant(1),
                           lambda: tf.constant(h * w))

        time = tf.constant(0)

        def body(time, outputs_ta, states_ta):
            constant_val = tf.constant(0)
            stateUp = tf.cond(tf.less_equal(tf.constant(w), time),
                              lambda: states_ta.read(getindex1(time, w)),
                              lambda: states_ta.read(h * w))
            stateLast = tf.cond(tf.less(constant_val, tf.mod(time, tf.constant(w))),
                                lambda: states_ta.read(getindex2(time, w)),
                                lambda: states_ta.read(h * w))

            currentState = stateUp[0], stateLast[0], stateUp[1], stateLast[1]
            out, state = cell(inputs_ta.read(time), currentState)
            outputs_ta = outputs_ta.write(time, out)
            states_ta = states_ta.write(time, state)
            return time + 1, outputs_ta, states_ta

        def condition(time, outputs_ta, states_ta):
            return tf.less(time, tf.constant(h * w))

        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations=1)

        outputs = outputs_ta.stack()
        states = states_ta.stack()

        y = tf.reshape(outputs, [h, w, batch_size, rnn_size])
        y = tf.transpose(y, [2, 0, 1, 3])
        if dims is not None:
            y = tf.reverse(y, dims)

        return y, states
