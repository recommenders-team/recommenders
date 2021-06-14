# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["NextItNetModel"]


class NextItNetModel(SequentialBaseModel):
    """NextItNet Model

    :Citation:
        Yuan, Fajie, et al. "A Simple Convolutional Generative Network
        for Next Item Recommendation", in Web Search and Data Mining, 2019.

    .. note::

        It requires strong sequence with dataset.
    """

    def _build_seq_graph(self):
        """The main function to create nextitnet model.

        Returns:
            obj: The output of nextitnet section.
        """
        hparams = self.hparams
        is_training = tf.equal(self.is_train_stage, True)
        item_history_embedding = tf.cond(
            is_training,
            lambda: self.item_history_embedding[:: self.hparams.train_num_ngs + 1],
            lambda: self.item_history_embedding,
        )
        cate_history_embedding = tf.cond(
            is_training,
            lambda: self.cate_history_embedding[:: self.hparams.train_num_ngs + 1],
            lambda: self.cate_history_embedding,
        )

        with tf.variable_scope("nextitnet", reuse=tf.AUTO_REUSE):

            dilate_input = tf.concat(
                [item_history_embedding, cate_history_embedding], 2
            )

            for layer_id, dilation in enumerate(hparams.dilations):
                dilate_input = tf.cond(
                    is_training,
                    lambda: self._nextitnet_residual_block_one(
                        dilate_input,
                        dilation,
                        layer_id,
                        dilate_input.get_shape()[-1],
                        hparams.kernel_size,
                        causal=True,
                        train=True,
                    ),
                    lambda: self._nextitnet_residual_block_one(
                        dilate_input,
                        dilation,
                        layer_id,
                        dilate_input.get_shape()[-1],
                        hparams.kernel_size,
                        causal=True,
                        train=False,
                    ),
                )

            self.dilate_input = dilate_input
            model_output = tf.cond(
                is_training, self._training_output, self._normal_output
            )

            return model_output

    def _training_output(self):
        model_output = tf.repeat(
            self.dilate_input, self.hparams.train_num_ngs + 1, axis=0
        )
        model_output = tf.concat([model_output, self.target_item_embedding], -1)
        model_output = tf.reshape(
            model_output,
            (
                -1,
                self.hparams.train_num_ngs + 1,
                self.hparams.max_seq_length,
                model_output.get_shape()[-1],
            ),
        )
        model_output = tf.transpose(model_output, [0, 2, 1, 3])
        model_output = tf.reshape(model_output, (-1, model_output.get_shape()[-1]))
        return model_output

    def _normal_output(self):
        model_output = self.dilate_input[:, -1, :]
        model_output = tf.concat(
            [model_output, self.target_item_embedding[:, -1, :]], -1
        )
        return model_output

    def _nextitnet_residual_block_one(
        self,
        input_,
        dilation,
        layer_id,
        residual_channels,
        kernel_size,
        causal=True,
        train=True,
    ):
        """The main function to use dilated CNN and residual network at sequence data

        Args:
            input_ (obj): The output of history sequential embeddings
            dilation (int): The dilation number of CNN layer
            layer_id (str): String value of layer ID, 0, 1, 2...
            residual_channels (int): Embedding size of input sequence
            kernel_size (int): Kernel size of CNN mask
            causal (bool): Whether to pad in front of the sequence or to pad surroundingly
            train (bool): is in training stage

        Returns:
            obj: The output of residual layers.
        """
        resblock_type = "decoder"
        resblock_name = "nextitnet_residual_block_one_{}_layer_{}_{}".format(
            resblock_type, layer_id, dilation
        )
        with tf.variable_scope(resblock_name):
            input_ln = self._layer_norm(input_, name="layer_norm1", trainable=train)
            relu1 = tf.nn.relu(input_ln)
            conv1 = self._conv1d(
                relu1, int(0.5 * int(residual_channels)), name="conv1d_1"
            )
            conv1 = self._layer_norm(conv1, name="layer_norm2", trainable=train)
            relu2 = tf.nn.relu(conv1)

            dilated_conv = self._conv1d(
                relu2,
                int(0.5 * int(residual_channels)),
                dilation,
                kernel_size,
                causal=causal,
                name="dilated_conv",
            )

            dilated_conv = self._layer_norm(
                dilated_conv, name="layer_norm3", trainable=train
            )
            relu3 = tf.nn.relu(dilated_conv)
            conv2 = self._conv1d(relu3, residual_channels, name="conv1d_2")
            return input_ + conv2

    def _conv1d(
        self,
        input_,
        output_channels,
        dilation=1,
        kernel_size=1,
        causal=False,
        name="dilated_conv",
    ):
        """Call a dilated CNN layer

        Returns:
            obj: The output of dilated CNN layers.
        """
        with tf.variable_scope(name):
            weight = tf.get_variable(
                "weight",
                [1, kernel_size, input_.get_shape()[-1], output_channels],
                initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1),
            )
            bias = tf.get_variable(
                "bias", [output_channels], initializer=tf.constant_initializer(0.0)
            )

            if causal:
                padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
                padded = tf.pad(input_, padding)
                input_expanded = tf.expand_dims(padded, dim=1)
                out = (
                    tf.nn.atrous_conv2d(
                        input_expanded, weight, rate=dilation, padding="VALID"
                    )
                    + bias
                )
            else:
                input_expanded = tf.expand_dims(input_, dim=1)
                out = (
                    tf.nn.conv2d(
                        input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME"
                    )
                    + bias
                )

            return tf.squeeze(out, [1])

    # tf.contrib.layers.layer_norm
    def _layer_norm(self, x, name, epsilon=1e-8, trainable=True):
        """Call a layer normalization

        Returns:
            obj: Normalized data
        """
        with tf.variable_scope(name):
            shape = x.get_shape()
            beta = tf.get_variable(
                "beta",
                [int(shape[-1])],
                initializer=tf.constant_initializer(0),
                trainable=trainable,
            )
            gamma = tf.get_variable(
                "gamma",
                [int(shape[-1])],
                initializer=tf.constant_initializer(1),
                trainable=trainable,
            )

            mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

            x = (x - mean) / tf.sqrt(variance + epsilon)

            return gamma * x + beta
