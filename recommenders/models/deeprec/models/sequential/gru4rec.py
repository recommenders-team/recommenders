# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from keras.layers.legacy_rnn.rnn_cell_impl import GRUCell, LSTMCell
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.compat.v1.nn import dynamic_rnn

__all__ = ["GRU4RecModel"]


class GRU4RecModel(SequentialBaseModel):
    """GRU4Rec Model

    :Citation:

        B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk, "Session-based Recommendations
        with Recurrent Neural Networks", ICLR (Poster), 2016.
    """

    def _build_seq_graph(self):
        """The main function to create GRU4Rec model.

        Returns:
            object:the output of GRU4Rec section.
        """
        with tf.compat.v1.variable_scope("gru4rec"):
            # final_state = self._build_lstm()
            final_state = self._build_gru()
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            tf.compat.v1.summary.histogram("model_output", model_output)
            return model_output

    def _build_lstm(self):
        """Apply an LSTM for modeling.

        Returns:
            object: The output of LSTM section.
        """
        with tf.compat.v1.name_scope("lstm"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
            self.history_embedding = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            rnn_outputs, final_state = dynamic_rnn(
                LSTMCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="lstm",
            )
            tf.compat.v1.summary.histogram("LSTM_outputs", rnn_outputs)
            return final_state[1]

    def _build_gru(self):
        """Apply a GRU for modeling.

        Returns:
            object: The output of GRU section.
        """
        with tf.compat.v1.name_scope("gru"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
            self.history_embedding = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            rnn_outputs, final_state = dynamic_rnn(
                GRUCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru",
            )
            tf.compat.v1.summary.histogram("GRU_outputs", rnn_outputs)
            return final_state
