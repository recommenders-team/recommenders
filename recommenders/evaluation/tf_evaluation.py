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


def accuracy(rating_true, rating_pred):
    """Accuracy using Tensorflow

    Evaluates accuracy evaluated on the rated items only (rated items are the ones with non-zero ratings).

    :math:`accuracy = 1/m \\sum_{mu=1}^{m} \\sum{i=1}^Nv 1/s(i) I(rating_true - rating_pred = 0)_{mu,i}`

    where `m = Nusers`, `Nv = number of items = number of visible units` and `s(i)` is the number of non-zero elements
    per row.

    Args:
        rating_true (tf.Tensor, float32): True Data.
        rating_pred (tf.Tensor, float32): Predicted Data.

    Returns:
        tf.Tensor: accuracy.

    """

    with tf.compat.v1.name_scope("accuracy"):

        # define and apply the mask
        mask = tf.not_equal(rating_true, 0)
        n_values = tf.reduce_sum(input_tensor=tf.cast(mask, dtype="float32"), axis=1)

        # Take the difference between the input data and the inferred ones. This value is zero whenever
        # the two values coincides
        vd = tf.compat.v1.where(
            mask, x=tf.abs(tf.subtract(rating_true, rating_pred)), y=tf.ones_like(rating_true)
        )

        # correct values: find the location where rating_true = rating_pred
        corr = tf.cast(tf.equal(vd, 0), dtype="float32")

        # evaluate accuracy
        accuracy_score = tf.reduce_mean(
            input_tensor=tf.compat.v1.div(
                tf.reduce_sum(input_tensor=corr, axis=1), n_values
            )
        )

    return accuracy_score
