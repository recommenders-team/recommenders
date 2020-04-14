# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.deeprec.models.base_modelv2 import BaseModelV2
from reco_utils.recommender.deeprec.models.layers import (
    MultiHeadAttention,
    AdditiveAttention,
)

__all__ = ["NRMSModel"]


class NRMSModel(BaseModelV2):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference 
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference 
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train(obj): NRMS data loader class for train data.
            iterator_creator_test(obj): NRMS data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("embedding"):
                self.embedding = tf.Variable(
                    tf.constant(
                        0.0,
                        shape=[hparams.word_size, hparams.word_emb_dim],
                        dtype=tf.float32,
                    ),
                    trainable=True,
                    name="word",
                )

                word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
                self.init_embedding = self.embedding.assign(word2vec_embedding)

        super().__init__(hparams, iterator_creator, graph=self.graph)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.
        
        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            np.array: A constant numpy array.
        """
        return tf.constant(np.load(file_path).astype(np.float32))

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams
        with tf.variable_scope("NRMS", initializer=self.initializer) as scope:
            train_logit, test_logit = self._build_nrms()
            return train_logit, test_logit

    def _build_userencoder(self):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder(obj): the news encoder of NRMS. 

        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        train_clicked_title = tf.reshape(
            self.train_iterator.clicked_news_batch, (-1, hparams.doc_size)
        )
        test_clicked_title = tf.reshape(
            self.test_iterator.clicked_news_batch, (-1, hparams.doc_size)
        )
        train_clicked_embedding = tf.nn.embedding_lookup(
            self.embedding, train_clicked_title
        )
        test_clicked_embedding = tf.nn.embedding_lookup(
            self.embedding, test_clicked_title
        )
        train_clicked_repr = self.sa_aggregate(
            train_clicked_embedding,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="title_encoder",
            dropout=hparams.dropout[0],
        )
        train_clicked_repr = tf.reshape(
            train_clicked_repr,
            (-1, hparams.his_size, hparams.head_dim * hparams.head_num),
        )
        test_clicked_repr = self.sa_aggregate(
            test_clicked_embedding,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="title_encoder",
            dropout=hparams.dropout[0],
        )
        test_clicked_repr = tf.reshape(
            test_clicked_repr,
            (-1, hparams.his_size, hparams.head_dim * hparams.head_num),
        )
        train_user_repr = self.sa_aggregate(
            train_clicked_repr,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="user_encoder",
            dropout=hparams.dropout[0],
        )
        test_user_repr = self.sa_aggregate(
            test_clicked_repr,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="user_encoder",
            dropout=hparams.dropout[0],
        )

        return train_user_repr, test_user_repr

    def _build_newsencoder(self):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        train_title_embedding = tf.nn.embedding_lookup(
            self.embedding,
            tf.reshape(
                self.train_iterator.candidate_news_batch, (-1, hparams.doc_size)
            ),
        )
        test_title_embedding = tf.nn.embedding_lookup(
            self.embedding,
            tf.reshape(self.test_iterator.candidate_news_batch, (-1, hparams.doc_size)),
        )
        train_title_repr = self.sa_aggregate(
            train_title_embedding,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="title_encoder",
            dropout=hparams.dropout[0],
        )
        train_title_repr = tf.reshape(
            train_title_repr,
            (-1, hparams.train_num_ngs + 1, hparams.head_dim * hparams.head_num),
        )
        test_title_repr = self.sa_aggregate(
            test_title_embedding,
            head_num=hparams.head_num,
            size_per_head=hparams.head_dim,
            name="title_encoder",
            dropout=hparams.dropout[0],
        )
        return train_title_repr, test_title_repr

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams

        self.train_title_repr, self.test_title_repr = self._build_newsencoder()
        self.train_user_repr, self.test_user_repr = self._build_userencoder()

        train_logit = layers.Dot(axes=-1)([self.train_title_repr, self.train_user_repr])
        test_logit = layers.Dot(axes=-1)([self.test_title_repr, self.test_user_repr])
        return train_logit, test_logit

    def _inner_product(self, src_vtx, dst_vtx):
        element_dot = tf.multiply(src_vtx, dst_vtx)
        reduce_sum = tf.reduce_sum(element_dot, axis=-1)
        return reduce_sum

    def sa_aggregate(self, word, name, head_num, size_per_head, dropout):
        hparams = self.hparams
        with tf.variable_scope(
            name_or_scope=name, default_name="sa_agg", reuse=tf.AUTO_REUSE, initializer=self.initializer
        ) as scope:
            layer_dropout = tf.layers.dropout(
                word, rate=dropout, training=self.is_train_stage
            )
            layer_sa = MultiHeadAttention(
                layer_dropout,
                layer_dropout,
                layer_dropout,
                nb_head=head_num,
                size_per_head=size_per_head,
                name="multi_head_attention",
            )
            layer_dropout = tf.layers.dropout(
                layer_sa, rate=dropout, training=self.is_train_stage
            )
            output = AdditiveAttention(layer_dropout, dim=hparams.attention_hidden_dim)
        return output
