# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["ASVDModel"]


class ASVDModel(SequentialBaseModel):
    def _build_seq_graph(self):
        """The main function to create ASVD model.
        
        Returns:
            obj:the output of ASVD section.
        """
        hparams = self.hparams
        with tf.variable_scope("asvd"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            with tf.variable_scope("Attention_layer"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                asvd_output = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("asvd_output", asvd_output)
            model_output = tf.concat([asvd_output, self.target_item_embedding], 1)
            self.model_output = model_output
            tf.summary.histogram("model_output", model_output)
            return model_output
