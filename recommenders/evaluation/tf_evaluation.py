# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf


def rmse(rating_true, rating_pred):
    """Root Mean Square Error using Tensorflow

    Note that this needs to be evaluated on the rated items only

    Args:
        rating_true (tf.Tensor, float32): True Data.
        rating_pred (tf.Tensor, float32): Predicted Data.

    Returns:
        tf.Tensor: root mean square error.

    """

    with tf.compat.v1.name_scope("re"):

        mask = tf.not_equal(rating_true, 0)  # selects only the rated items
        n_values = tf.reduce_sum(
            input_tensor=tf.cast(mask, dtype="float32"), axis=1
        )  # number of rated items

        # evaluate the square difference between the inferred and the input data on the rated items
        e = tf.compat.v1.where(
            mask, x=tf.math.squared_difference(rating_true, rating_pred), y=tf.zeros_like(rating_true)
        )

        # evaluate the msre
        err = tf.sqrt(
            tf.reduce_mean(
                input_tensor=tf.compat.v1.div(
                    tf.reduce_sum(input_tensor=e, axis=1), n_values
                )
            )
            / 2
        )

    return err
