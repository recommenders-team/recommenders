# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["A2SVDModel"]


class A2SVDModel(SequentialBaseModel):
    """A2SVD Model (Attentive Asynchronous Singular Value Decomposition)

    It extends ASVD with an attention module.

    :Citation:

        ASVD: Y. Koren, "Factorization Meets the Neighborhood: a Multifaceted Collaborative
        Filtering Model", in Proceedings of the 14th ACM SIGKDD international conference on
        Knowledge discovery and data mining, pages 426–434, ACM, 2008.

        A2SVD: Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
        Long and Short-Term Preferences for Personailzed Recommendation", in Proceedings of
        the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19,
        Pages 4213-4219, AAAI Press, 2019.
    """

    def _build_seq_graph(self):
        """The main function to create A2SVD model.

        Returns:
            object: The output of A2SVD section.
        """
        hparams = self.hparams
        with tf.variable_scope("a2svd"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            with tf.variable_scope("Attention_layer"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                asvd_output = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("a2svd_output", asvd_output)
            model_output = tf.concat([asvd_output, self.target_item_embedding], 1)
            self.model_output = model_output
            tf.summary.histogram("model_output", model_output)
            return model_output
