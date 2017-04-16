# -*- coding: utf-8 -*-
'''
Tensorflow-Related utilities.

Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import absolute_import, division, print_function
from biutils.misc import generate_slices
from scipy import sparse
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.contrib import slim

def generate_minibatches(batch_size, x_placeholder, y_placeholder,
                         x_data, y_data, n_epochs=1,
                         ignore_last_minibatch_if_smaller=True, shuffle=True,
                         feed_dict=None):
    cnt_epochs = 0
    while True:
        if shuffle:
            idx = np.arange(x_data.shape[0])
            np.random.shuffle(idx)
            x_data, y_data = x_data[idx], y_data[idx]

        if feed_dict is None:
                feed_dict = {}

        for s in generate_slices(x_data.shape[0], batch_size,
                                 ignore_last_minibatch_if_smaller):
            xx, yy = x_data[s], y_data[s]
            if sparse.issparse(xx):
                xx = xx.A

            feed_dict[x_placeholder] = xx
            feed_dict[y_placeholder] = yy
            yield feed_dict
        cnt_epochs += 1
        if n_epochs is not None and cnt_epochs >= n_epochs:
            break


def track_activation_summary(x, weights=None, bias=None):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

    if weights is not None:
        tf.histogram_summary(x.op.name + '/weights', weights)
    if bias is not None:
        tf.histogram_summary(x.op.name + '/bias', bias)


def create_training_op(loss, learning_rate, momentum, global_step,
                       add_summaries=True):
    tf.scalar_summary('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    grads = optimizer.compute_gradients(loss)
    for grad, var in grads:
        if grad is not None and add_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = optimizer.apply_gradients(grads,
                                                  global_step=global_step)
    return apply_gradient_op


def get_ce_loss(logits, y, summary_prefix='', add_summaries=True):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, y, name='cross_entropy_per_example')
    ce_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    if add_summaries:
        tf.scalar_summary(summary_prefix+'cross_entropy', ce_mean)
    if len(tf.get_collection('regularization_losses')) > 0:
        loss = ce_mean + tf.add_n(tf.get_collection('regularization_losses'))
        if add_summaries:
            tf.scalar_summary(summary_prefix+'loss', loss)
    else:
        loss = ce_mean
    return loss


def get_multitask_ce_loss(logits, y, summary_prefix='', add_summaries=True):
    '''Expects invalid labels to be encoded as -1.'''
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, y, name='cross_entropy_per_example')
    mask = tf.cast(tf.not_equal(y, -1), tf.float32)
    cross_entropy *= mask
    ce_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    if add_summaries:
        tf.scalar_summary(summary_prefix+'cross_entropy', ce_mean)
    if len(tf.get_collection('regularization_losses')) > 0:
        loss = ce_mean + tf.add_n(tf.get_collection('regularization_losses'))
        if add_summaries:
            tf.scalar_summary(summary_prefix+'loss', loss)
    else:
        loss = ce_mean
    return loss


def add_linear_layer(x, n_outputs, keep_prob=None, activation_fn=tf.identity,
                     w_initializer=None, regularizer=None, b_initializer=None,
                     add_summaries=True):
    if w_initializer is None:
        w_initializer = layers.initializers.variance_scaling_initializer()
    n_hiddens = x.get_shape()[-1]
    w = tf.get_variable("weights", [n_hiddens, n_outputs],
                        initializer=w_initializer)
    b = tf.get_variable('bias', [n_outputs], initializer=b_initializer)
    x = tf.matmul(x, w) + b
    x = activation_fn(x)
    if keep_prob is not None:
        x = tf.nn.dropout(x, keep_prob)
    if add_summaries:
        track_activation_summary(x, w, b)
    return x


def add_conv_layer(x, kernel_shape, n_filters, strides=[1, 1],
                   activation_fn=tf.nn.relu, initializer=None, regularizer=None, add_summaries=True):
    k = [kernel_shape[0], kernel_shape[1], x.get_shape()[-1], n_filters]
    if initializer is None:
        initializer = layers.initializers.variance_scaling_initializer()
    w = tf.get_variable('weights', k, initializer=initializer, regularizer=regularizer)
    b = tf.get_variable('bias', [n_filters], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, w, [1, strides[0], strides[1], 1], padding='SAME')
    conv = activation_fn(conv)
    if add_summaries:
        track_activation_summary(conv, w, b)
    return conv


def resize_and_crop(image, smaller_size, cropped_size):
    '''
    Resizes the Image as in Krizhevsky et. al 2012:
    Resizes an image such that the smaller side is then smaller_size pixels
    long and pads the other side to retain aspect ratio,
    then crops out the central cropped_size*cropped_size image patch.
    '''
    img_shape = tf.shape(image)
    oh = img_shape[0]
    ow = img_shape[1]
    s = tf.maximum(oh, ow) /  tf.minimum(oh, ow)
    smaller_size = tf.constant(smaller_size, tf.float64)

    bigger_size = tf.cast(tf.mul(s, smaller_size), tf.int32)
    smaller_size = tf.cast(smaller_size, tf.int32)

    th = tf.cond(oh < ow, lambda: smaller_size, lambda: bigger_size)
    tw = tf.cond(oh < ow, lambda: bigger_size, lambda: smaller_size)

    image = tf.image.resize_images(image, th, tw)  # rescale
    image = tf.image.resize_image_with_crop_or_pad(image, cropped_size, cropped_size)
    image.set_shape((cropped_size, cropped_size, 3))
    return image



from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import errors
import threading
class MyRunnerBase(tf.train.QueueRunner):
    def __init__(self, outputs, batch_size, n_threads, shuffle=True, capacity=None, enqueue_many=False,
                 shapes=None, dtypes=None, name=None):
        self.n_threads = n_threads
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.outputs = outputs
        self._name = name
        if shapes is None:
            shapes = [o.get_shape().as_list() for o in self.outputs]
        if dtypes is None:
            dtypes = [o.dtype for o in self.outputs]
        if capacity is None:
            capacity = self.batch_size*(self.n_threads+1)
        if shuffle == True:
            min_cap = capacity // 4
            self._queue = tf.RandomShuffleQueue(shapes=shapes, dtypes=dtypes,
                                                min_after_dequeue=min_cap,
                                                capacity=capacity)
        else:
            self._queue = tf.FIFOQueue(shapes=shapes, dtypes=dtypes, capacity=capacity)

        if enqueue_many:
            self.enqueue_op = self._queue.enqueue_many(self.outputs)
        else:
            self.enqueue_op = self._queue.enqueue(self.outputs)

        super(MyRunnerBase, self).__init__(self._queue, [self.enqueue_op]*n_threads)
        if name is not None:
            c = tf.cast(self._queue.size(), tf.float32) * (1. / capacity)
            self._capacity_summary = tf.summary.scalar("%s_fullness" % name, c)


    def get_data(self):
        ''' Return's tensors containing a batch of images and labels. '''
        batch = self._queue.dequeue_many(self.batch_size)
        return batch

    def _setup_thread(self):
        '''
        Sets up each thread. This can be used to allocate any thread-local
        data (such as a copy of the input data for each thread).
        '''
        return None

    def _prepare_epoch(self, thread_local_storage):
        return thread_local_storage

    def _run(self, sess, enqueue_op=None, coord=None):
        """Execute the enqueue op in a loop, close the queue in case of error.
        Args:
            sess: A Session.
            enqueue_op: The Operation to run.
            coord: Optional Coordinator object for reporting errors and checking
                for stop conditions.
        """
        if coord:
            coord.register_thread(threading.current_thread())

        thread_local_data = self._setup_thread()
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break

                thread_local_data = self._prepare_epoch(thread_local_data)
                try:
                    self._enqueue_epoch(sess, thread_local_data, coord)
                except self._queue_closed_exception_types:
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except tf.errors.CancelledError as e:  # when we are requested to stop
            if coord:
                coord.request_stop(e)
            else:
                raise
        except Exception as e:
            # This catches all other exceptions.
            print("Exception in MyRunnerBase: %s" % str(e), flush=True)
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s" % str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1


    def create_threads(self, sess, coord=None, daemon=False, start=False):
        """Create threads to run the enqueue ops.
        This method requires a session in which the graph was launched. It creates
        a list of threads, optionally starting them. There is one thread for each
        op passed in `enqueue_ops`.
        The `coord` argument is an optional coordinator, that the threads will use
        to terminate together and report exceptions. If a coordinator is given,
        this method starts an additional thread to close the queue when the
        coordinator requests a stop.
        This method may be called again as long as all threads from a previous call
        have stopped.
        Args:
          sess: A `Session`.
          coord: Optional `Coordinator` object for reporting errors and checking
                 stop conditions.
          daemon: Boolean. If `True` make the threads daemon threads.
          start: Boolean. If `True` starts the threads. If `False` the
                 caller must call the `start()` method of the returned threads.
        Returns:
          A list of threads.
        Raises:
          RuntimeError: If threads from a previous call to `create_threads()` are
          still running.
        """
        with self._lock:
            try:
                if self._runs_per_session[sess] > 0:
                    # Already started: no new threads to return.
                    return []
            except KeyError:
                # We haven't seen this session yet.
                pass
            self._runs_per_session[sess] = len(self._enqueue_ops)
            self._exceptions_raised = []
            self._runs = self.n_threads

        ret_threads = [threading.Thread(target=self._run, args=(sess, None, coord))
                       for i in range(self.n_threads)]
        if coord:
            ret_threads.append(threading.Thread(target=self._close_on_stop,
                               args=(sess, self._cancel_op, coord)))
        for t in ret_threads:
            if coord:
                coord.register_thread(t)
            if daemon:
                t.daemon = True
            if start:
                t.start()
        return ret_threads


class NumpyQueueRunner(MyRunnerBase):
    """
    Enqueues data from numpy arrays after preprocessing in a thread.

    This is esentially a QueueRunner, but it takes its data from numpy arrays
    (e.g. a list of filenames) instead of tensorflow constructs (like an
    InputStringProducer).
    This is usually faster (it saves the whole queuing from the input string
    producer).
    This class is based on
        https://indico.io/blog/tensorflow-data-input-part2-extensions/
    So see there for more details.
    """
    def __init__(self, input_list, preprocess_function,
                 batch_size, n_threads, shuffle=True, capacity=None):
        self.inputs = np.array(input_list)
        self.placeholders = []
        for i in range(len(input_list)):
            p = tf.placeholder(dtype=tf.string, shape=[], name='input_%d'%i)
            self.placeholders.append(p)
        outputs = preprocess_function(*self.placeholders)
        super(NumpyQueueRunner, self).__init__(outputs, batch_size, n_threads,
                                               shuffle, capacity)

    def _setup_thread(self):
        with self._lock:
            a = self.inputs.copy()
        return a

    def _prepare_epoch(self, a):
        if self.shuffle:
            idx = np.arange(a.shape[1])
            np.random.shuffle(idx)
            a = a[:, idx]
        return a

    def _enqueue_epoch(self, sess, a, coord):
        n_entries = len(a[0])
        n_inputs = len(self.inputs)
        for i in range(n_entries):
            tmp = [(self.placeholders[j], a[j][i]) for j in range(n_inputs)]
            feed_dict = dict(tmp)
            sess.run(self.enqueue_op, feed_dict=feed_dict)
            if coord and coord.should_stop():
                break


def produce_minibatches(input_list, batch_size, preprocess_function,
                        n_threads, shuffle=True, capacity=None):
    qr = NumpyQueueRunner(input_list, preprocess_function, batch_size,
                          n_threads, shuffle, capacity)
    tf.train.add_queue_runner(qr)
    return qr.get_data()


_MyLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "i", "f", "o"))

class MyLSTMStateTuple(_MyLSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, i, f, o) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype


class LSTMwithSummaries(tf.contrib.rnn.RNNCell):
    """Basic LSTM recurrent network cell that has summaries for interesting things.
    """

    def __init__(self, num_units, i_init=1.0, f_init=0.0, o_init=0.0, activation=tf.nn.tanh):
        """Initialize the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            activation: Activation function of the inner states.
        """
        self._num_units = num_units
        self._activation = activation
        self._bias_inits = (i_init, 0.0, f_init, o_init)
        self._summary_vars = {}

    def create_summaries(self, state):
        c, h, i, f, o = state
        self._summary_vars.update({"act_i": i, "act_f": f,
            "act_o": o, "act_c": c, "act_h": h})

        s = [tf.summary.histogram(k, v) for (k, v) in self._summary_vars.items()]
        return tf.summary.merge(s)

    @property
    def state_size(self):
        n = self._num_units
        return MyLSTMStateTuple(n, n, n, n, n)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h, i, f, o = state
            concat = self._linear(inputs, h)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = (c * f + i * self._activation(j))
            new_h = self._activation(new_c) * o
            new_state = MyLSTMStateTuple(new_c, new_h, i, f, o)
            return new_h, new_state

    def _linear(self, x, h, scope=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
        Args:
            args: a 2D Tensor or a list of 2D, batch x n, Tensors.
            output_size: int, second dimension of W[i].
            bias: boolean, whether to add a bias term or not.
            bias_start: starting value to initialize the bias; 0 by default.
        Returns:
            A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
        """

        args = tf.concat(1, [x, h])
        dt = args.dtype
        xs = args.get_shape()[1]
        nh = self._num_units
        with tf.variable_scope(scope or "Linear"):
            #winit = tf.truncated_normal_initializer(stddev=0.001)
            winit = slim.initializers.xavier_initializer()
            weights = tf.get_variable("weights", [xs, 4*nh], dtype=dt, initializer=winit)

            inits = [tf.constant_initializer(b, dtype=dt) for b in self._bias_inits]
            ib = tf.get_variable("i_bias", [self._num_units], dtype=dt, initializer=inits[0])
            jb = tf.get_variable("j_bias", [self._num_units], dtype=dt, initializer=inits[1])
            fb = tf.get_variable("f_bias", [self._num_units], dtype=dt, initializer=inits[2])
            ob = tf.get_variable("o_bias", [self._num_units], dtype=dt, initializer=inits[3])
            bias = tf.concat(0, [ib, jb, fb, ob])

        res = tf.matmul(args, weights) + bias
        self._summary_vars.update({"bias_i": ib, "bias_f": fb, "bias_o": ob, 'w': weights})
        return res


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


import numbers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
def dropout_selu(x, rate, alpha=-1.7580993408473766, noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        #binary_tensor2 = math_ops.ceil(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        #a = tf.sqrt(1.0/(keep_prob+alpha^2*keep_prob*(1.0-keep_prob)))
        a = tf.sqrt(1.0 / keep_prob + tf.pow(alpha,2) * keep_prob * 1.0 - keep_prob)
        #a = tf.sqrt(tf.div(1.0, tf.add(keep_prob ,tf.multiply(tf.pow(alpha,2) , tf.multiply(keep_prob,    tf.subtract(1.0,keep_prob)))) ))

        b = -a * (1 - keep_prob) * alpha
        #b = tf.neg( tf.mul(a , (tf.multiply(tf.subtract(1.0,keep_prob),alpha))))
        ret = a * ret + b
        #ret = tf.add(tf.multiply(a , ret) , b)
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))


def saltpepper_noise(x, noise_rate, ones_rate=0.5, training=False, name=None):
    ''' Adds saltpepper noise (sets some elements to either 0 or 1).
        ones_rate controls the fraction of pepper (1s) vs salt (0s).
        ones_rate == 0 =>  dropout (noise value is always 0)
        ones_rate == 1 => dropin (noise value is always 1)
    '''

    def saltpepper_noise_impl(x, noise_rate, ones_rate):
        assert 0 <= noise_rate <= 1
        assert 0 <= ones_rate <= 1
        b = tf.floor(tf.random_uniform(x.get_shape(), 0, 1)  + ones_rate)
        c = tf.random_uniform(x.get_shape(), 0, 1) < noise_rate
        return tf.where(c, b, x)

    with ops.name_scope(name, "inputnoise", [x]) as name:
        return utils.smart_cond(training,
            lambda: saltpepper_noise_impl(x, noise_rate, ones_rate),
            lambda: array_ops.identity(x))
