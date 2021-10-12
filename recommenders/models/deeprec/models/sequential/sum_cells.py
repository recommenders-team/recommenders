# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class SUMCell(LayerRNNCell):
    """Cell for Sequential User Matrix"""

    def __init__(
        self,
        num_units,
        slots,
        attention_size,
        input_size,
        activation=None,
        reuse=None,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        dtype=None,
        **kwargs
    ):
        super(SUMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.",
                self,
            )

        self._input_size = input_size
        self._slots = slots - 1  ## the last channel is reserved for the highway slot
        self._num_units = num_units
        self._real_units = (self._num_units - input_size) // slots
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _basic_build(self, inputs_shape):
        """Common initialization operations for SUM cell and its variants.
        This function creates parameters for the cell.
        """

        d = inputs_shape[-1]
        h = self._real_units
        s = self._slots

        self._erase_W = self.add_variable(
            name="_erase_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._erase_b = self.add_variable(
            name="_erase_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._reset_W = self.add_variable(
            name="_reset_W", shape=[d + h, 1], initializer=self._kernel_initializer
        )
        self._reset_b = self.add_variable(
            name="_reset_b",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._add_W = self.add_variable(
            name="_add_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._add_b = self.add_variable(
            name="_add_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self.heads = self.add_variable(
            name="_heads", shape=[s, d], initializer=self._kernel_initializer
        )

        self._beta = self.add_variable(
            name="_beta_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(np.array([1.02]), dtype=np.float32),
        )
        self._alpha = self.add_variable(
            name="_alpha_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(np.array([0.98]), dtype=np.float32),
        )

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUM cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units
        s = self._slots

        self._basic_build(inputs_shape)

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUM cell to process user behaviors.

        params:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        returns:
            state, state: 
            - after process the user behavior at time T, returns (a batch of) new user states at time T
            - after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1
        state, last = state[:, : s * h], state[:, s * h :]
        state = tf.reshape(state, [-1, s, h])

        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)

        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1)
        h_hat = (h_hat + state[:, self._slots, :]) / 2

        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]
        state01 = start_part01 * (
            tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, inputs], axis=-1)
        return state, state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(SUMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SUMV2Cell(SUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units
        s = self._slots

        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1
        state, last = state[:, : s * h], state[:, s * h :]
        state = tf.reshape(state, [-1, s, h])

        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)

        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1)
        h_hat = (h_hat + state[:, self._slots, :]) / 2

        ## get the true writing attentions
        writing_input = tf.concat([inputs, h_hat], axis=1)
        att_weights = tf.compat.v1.nn.xw_plus_b(writing_input, self._writing_W, self._writing_b)
        att_weights = tf.nn.relu(att_weights)
        att_weights = tf.matmul(att_weights, self._writing_W02)
        att_weights = tf.nn.softmax(att_weights, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]
        state01 = start_part01 * (
            tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, inputs], axis=-1)
        return state, state


def _check_rnn_cell_input_dtypes(inputs):
    for t in nest.flatten(inputs):
        _check_supported_dtypes(t.dtype)


def _check_supported_dtypes(dtype):
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError(
            "RNN cell only supports floating point inputs, " "but saw dtype: %s" % dtype
        )
