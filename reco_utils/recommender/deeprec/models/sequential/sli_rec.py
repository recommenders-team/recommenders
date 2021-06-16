# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["SLI_RECModel"]


class SLI_RECModel(SequentialBaseModel):
    """SLI Rec model

    :Citation:

        Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
        Long and Short-Term Preferences for Personailzed Recommendation", in Proceedings of
        the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19,
        Pages 4213-4219, AAAI Press, 2019.
    """

    def _build_seq_graph(self):
        """The main function to create sli_rec model.

        Returns:
            object: the output of sli_rec section.
        """
        hparams = self.hparams
        with tf.variable_scope("sli_rec"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                att_fea1 = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("att_fea1", att_fea1)

            item_history_embedding_new = tf.concat(
                [
                    self.item_history_embedding,
                    tf.expand_dims(self.iterator.time_from_first_action, -1),
                ],
                -1,
            )
            item_history_embedding_new = tf.concat(
                [
                    item_history_embedding_new,
                    tf.expand_dims(self.iterator.time_to_now, -1),
                ],
                -1,
            )
            with tf.variable_scope("rnn"):
                rnn_outputs, final_state = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
                tf.summary.histogram("LSTM_outputs", rnn_outputs)

            with tf.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, rnn_outputs
                )
                att_fea2 = tf.reduce_sum(att_outputs2, 1)
                tf.summary.histogram("att_fea2", att_fea2)

            # ensemble
            with tf.name_scope("alpha"):
                concat_all = tf.concat(
                    [
                        self.target_item_embedding,
                        att_fea1,
                        att_fea2,
                        tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                    ],
                    1,
                )
                last_hidden_nn_layer = concat_all
                alpha_logit = self._fcn_net(
                    last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                )
                alpha_output = tf.sigmoid(alpha_logit)
                user_embed = att_fea1 * alpha_output + att_fea2 * (1.0 - alpha_output)
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

    def _attention_fcn(self, query, user_embedding):
        """Apply attention by fully connected layers.

        Args:
            query (object): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (object): The output of RNN layers which is regarded as user modeling.

        Returns:
            object: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn"):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output
