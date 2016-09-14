# -*- coding: utf-8 -*-
'''
Tensorflow-Related utilities.

Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import absolute_import, division, print_function
from biutils.misc import generate_slices

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def generate_minibatches(batch_size, x_placeholder, y_placeholder,
                         x_data, y_data, n_epochs=None,
                         ignore_last_minibatch_if_smaller=True, shuffle=True,
                         feed_dict=None):
    cnt_epochs = 0
    while True:
        if shuffle:
            idx = np.arange(x_data.shape[0])
            np.random.shuffle(idx)
            x_data, y_data = x_data[idx], y_data[idx]

        for s in generate_slices(x_data.shape[0], batch_size,
                                 ignore_last_minibatch_if_smaller):
            xx, yy = x_data[s], y_data[s]
            if feed_dict is None:
                feed_dict = {x_placeholder: xx, y_placeholder: yy}
            else:
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


# this is be part of tf.python.training.learning_rate_decay in tf from 0.10
def polynomial_decay(learning_rate, global_step, decay_steps,
                     end_learning_rate=0.0001, power=1.0,
                     cycle=False, name=None):
    with ops.op_scope(
      [learning_rate, global_step, decay_steps, end_learning_rate, power],
      name, "PolynomialDecay") as name:
        learning_rate = ops.convert_to_tensor(learning_rate,
                                              name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        end_learning_rate = math_ops.cast(end_learning_rate, dtype)
        power = math_ops.cast(power, dtype)
        if cycle:
            # Find the first multiple of decay_steps that is
            # bigger than global_step.
            c = math_ops.ceil(global_step / decay_steps)
            decay_steps = math_ops.mul(decay_steps, c)
        else:
            # Make sure that global_step used is not bigger than decay_steps.
            global_step = math_ops.minimum(global_step, decay_steps)

        p = math_ops.div(global_step, decay_steps)
        return math_ops.add(math_ops.mul(learning_rate - end_learning_rate,
                                         math_ops.pow(1 - p, power)),
                            end_learning_rate, name=name)



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
    image = resize_image_with_crop_or_pad(image, cropped_size, cropped_size)
    image.set_shape((cropped_size, cropped_size, 3))
    return image




#
# The following functions can't work if the image shape isn't set
# in TF < 0.10. Thus we can't use them if we don't know the size of the
# image in advance.
# TODO: delete these functions once we switch to TF >= 0.10
#
def resize_image_with_crop_or_pad(image, target_height, target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    zero = tf.constant(0)
    half = tf.constant(2)

    offset_crop_width = tf.python.control_flow_ops.cond(
        tf.less(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.sub(original_width, target_width), half),
        lambda: zero)

    offset_pad_width = tf.python.control_flow_ops.cond(
        tf.greater(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.sub(target_width, original_width), half),
        lambda: zero)

    offset_crop_height = tf.python.control_flow_ops.cond(
        tf.less(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.sub(original_height, target_height), half),
        lambda: zero)

    offset_pad_height = tf.python.control_flow_ops.cond(
        tf.greater(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.sub(target_height, original_height), half),
        lambda: zero)

    cropped = crop_to_bounding_box(
        image, offset_crop_height, offset_crop_width,
        tf.minimum(target_height, original_height),
        tf.minimum(target_width, original_width))

    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)

    return resized


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
    cropped = tf.slice(
        image,
        tf.pack([offset_height, offset_width, 0]),
        tf.pack([target_height, target_width, -1]))

    return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    after_padding_width = tf.sub(
        tf.sub(target_width, offset_width),  original_width)
    after_padding_height = tf.sub(
        tf.sub(target_height, offset_height), original_height)

    paddings = tf.reshape(
        tf.pack(
            [offset_height, after_padding_height,
             offset_width, after_padding_width,
             0, 0]), [3, 2])

    padded = tf.pad(image, paddings)

    return padded



from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import errors
import threading
class MyRunnerBase(tf.train.QueueRunner):
    def __init__(self, outputs, batch_size, n_threads, shuffle=True, capacity=None):
        self.n_threads = n_threads
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.outputs = outputs
        shapes = [o.get_shape().as_list() for o in self.outputs]
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

        self.enqueue_op = self.queue.enqueue(self.outputs)

        # QueueRunner implementation changed a bit after 0.9, so bring _run
        # up to the new implementation if you use tf 0.10
        assert tf.__version__ == '0.9.0', "Update to new TF API"

        super(MyRunnerBase, self).__init__(self._queue, [self.enqueue_op, ])


    def get_data(self):
        ''' Return's tensors containing a batch of images and labels. '''
        batch = self._queue.dequeue_many(self.batch_size)
        return batch

    def _setup_thread(self):
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
        thread_local_data = self._setup_thread()

        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break

                thread_local_data = self._prepare_epoch(thread_local_data)
                try:
                    self._enqueue_epoch(sess, thread_local_data, coord)
                except errors.OutOfRangeError:
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs -= 1
                        decremented = True
                        if self._runs == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs -= 1


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
            if self._runs > 0:
                # Already started: no new threads to return.
                return []
            self._runs = self.n_threads
            self._exceptions_raised = []

        ret_threads = [threading.Thread(target=self._run, args=(sess, None, coord))
                       for i in range(self.n_threads)]
        if coord:
            ret_threads.append(threading.Thread(target=self._close_on_stop,
                               args=(sess, self._cancel_op, coord)))
        for t in ret_threads:
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
