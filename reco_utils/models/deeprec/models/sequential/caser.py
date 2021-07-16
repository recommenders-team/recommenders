# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["CaserModel"]


class CaserModel(SequentialBaseModel):
    """Caser Model

    :Citation:

        J. Tang and K. Wang, "Personalized top-n sequential recommendation via convolutional
        sequence embedding", in Proceedings of the Eleventh ACM International Conference on
        Web Search and Data Mining, ACM, 2018.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables for caser

        Args:
            hparams (object): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
        """
        self.hparams = hparams
        self.L = hparams.L  # history sequence that involved in convolution shape
        self.T = hparams.T  # prediction shape
        self.n_v = hparams.n_v  # number of vertical convolution layers
        self.n_h = hparams.n_h  # number of horizonal convolution layers
        self.lengths = [
            i + 1 for i in range(self.L)
        ]  # horizonal convolution filter shape
        super().__init__(hparams, iterator_creator, seed=seed)

    def _build_seq_graph(self):
        """The main function to create caser model.

        Returns:
            object: The output of caser section.
        """
        with tf.variable_scope("caser"):
            cnn_output = self._caser_cnn()
            model_output = tf.concat([cnn_output, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

    def _add_cnn(self, hist_matrix, vertical_dim, scope):
        """The main function to use CNN at both vertical and horizonal aspects.

        Args:
            hist_matrix (object): The output of history sequential embeddings
            vertical_dim (int): The shape of embeddings of input
            scope (object): The scope of CNN input.

        Returns:
            object: The output of CNN layers.
        """
        with tf.variable_scope(scope):
            with tf.variable_scope("vertical"):
                embedding_T = tf.transpose(hist_matrix, [0, 2, 1])
                out_v = self._build_cnn(embedding_T, self.n_v, vertical_dim)
                out_v = tf.layers.flatten(out_v)
            with tf.variable_scope("horizonal"):
                out_hs = []
                for h in self.lengths:
                    conv_out = self._build_cnn(hist_matrix, self.n_h, h)
                    max_pool_out = tf.reduce_max(
                        conv_out, reduction_indices=[1], name="max_pool_{0}".format(h)
                    )
                    out_hs.append(max_pool_out)
                out_h = tf.concat(out_hs, 1)
        return tf.concat([out_v, out_h], 1)

    def _caser_cnn(self):
        """The main function to use CNN at both item and category aspects.

        Returns:
            object: The concatenated output of two parts of item and category.
        """
        item_out = self._add_cnn(
            self.item_history_embedding, self.item_embedding_dim, "item"
        )
        tf.summary.histogram("item_out", item_out)
        cate_out = self._add_cnn(
            self.cate_history_embedding, self.cate_embedding_dim, "cate"
        )
        tf.summary.histogram("cate_out", cate_out)
        cnn_output = tf.concat([item_out, cate_out], 1)
        tf.summary.histogram("cnn_output", cnn_output)
        return cnn_output

    def _build_cnn(self, history_matrix, nums, shape):
        """Call a CNN layer.

        Returns:
            object: The output of cnn section.
        """
        return tf.layers.conv1d(
            history_matrix,
            nums,
            shape,
            activation=tf.nn.relu,
            name="conv_" + str(shape),
        )
