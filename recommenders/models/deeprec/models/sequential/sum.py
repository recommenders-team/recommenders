# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.nn import dynamic_rnn
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from recommenders.models.deeprec.models.sequential.sum_cells import (
    SUMCell,
    SUMV2Cell,
)


class SUMModel(SequentialBaseModel):
    """Sequential User Matrix Model

    :Citation:

        Lian, J., Batal, I., Liu, Z., Soni, A., Kang, E. Y., Wang, Y., & Xie, X.,
        "Multi-Interest-Aware User Modeling for Large-Scale Sequential Recommendations", arXiv preprint arXiv:2102.09211, 2021.
    """

    def _build_seq_graph(self):
        """The main function to create SUM model.

        Returns:
            object: The output of SUM section, which is a concatenation of user vector and target item vector.
        """
        hparams = self.hparams
        with tf.variable_scope("sum"):
            self.history_embedding = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            cell = self._create_sumcell()
            self.cell = cell
            cell.model = self
            final_state = self._build_sum(cell)

            for _p in cell.parameter_set:
                tf.summary.histogram(_p.name, _p)
            if hasattr(cell, "_alpha") and hasattr(cell._alpha, "name"):
                tf.summary.histogram(cell._alpha.name, cell._alpha)
            if hasattr(cell, "_beta") and hasattr(cell._beta, "name"):
                tf.summary.histogram(cell._beta.name, cell._beta)

            final_state, att_weights = self._attention_query_by_state(
                final_state, self.target_item_embedding
            )
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
        return model_output

    def _attention_query_by_state(self, seq_output, query):
        """Merge a user's memory states conditioned by a query item.

        Params:
            seq_output: A flatten representation of SUM memory states for (a batch of) users
            query: (a batch of) target item candidates

        Returns:
            tf.Tensor, tf.Tensor: Merged user representation. Attention weights of each memory channel.
        """
        dim_q = query.shape[-1].value
        att_weights = tf.constant(1.0, dtype=tf.float32)
        with tf.variable_scope("query_att"):
            if self.hparams.slots > 1:
                query_att_W = tf.get_variable(
                    name="query_att_W",
                    shape=[self.hidden_size, dim_q],
                    initializer=self.initializer,
                )

                # reshape the memory states to (BatchSize, Slots, HiddenSize)
                memory_state = tf.reshape(
                    seq_output, [-1, self.hparams.slots, self.hidden_size]
                )

                att_weights = tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.tensordot(memory_state, query_att_W, axes=1),
                            tf.expand_dims(query, -1),
                        ),
                        -1,
                    ),
                    -1,
                )
                # merge the memory states, the final shape is (BatchSize, HiddenSize)
                att_res = tf.reduce_sum(
                    memory_state * tf.expand_dims(att_weights, -1), 1
                )

            else:
                att_res = seq_output

        return att_res, att_weights

    def _create_sumcell(self):
        """Create a SUM cell

        Returns:
            object: An initialized SUM cell
        """
        hparams = self.hparams
        input_embedding_dim = self.history_embedding.shape[-1]
        input_params = [
            hparams.hidden_size * hparams.slots + input_embedding_dim,
            hparams.slots,
            hparams.attention_size,
            input_embedding_dim,
        ]
        sumcells = {"SUM": SUMCell, "SUMV2": SUMV2Cell}
        sumCell = sumcells[hparams.cell]
        res = None
        if hparams.cell in ["SUM", "SUMV2"]:
            res = sumCell(*input_params)
        else:
            raise ValueError("ERROR! Cell type not support: {0}".format(hparams.cell))
        return res

    def _build_sum(self, cell):
        """Generate  user memory states from behavior sequence

        Args:
            object: An initialied SUM cell.

        Returns:
            object: A flatten representation of user memory states, in the shape of (BatchSize, SlotsNum x HiddenSize)
        """
        hparams = self.hparams
        with tf.variable_scope("sum"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            rum_outputs, final_state = dynamic_rnn(
                cell,
                inputs=self.history_embedding,
                dtype=tf.float32,
                sequence_length=self.sequence_length,
                scope="sum",
                initial_state=cell.zero_state(
                    tf.shape(self.history_embedding)[0], tf.float32
                ),
            )

            final_state = final_state[:, : hparams.slots * hparams.hidden_size]

            self.heads = cell.heads
            self.alpha = cell._alpha
            self.beta = cell._beta
            tf.summary.histogram("SUM_outputs", rum_outputs)

        return final_state
