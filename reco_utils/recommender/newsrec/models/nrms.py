# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference 
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference 
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
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

        super().__init__(hparams, iterator_creator, seed=seed)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.
        
        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            np.array: A constant numpy array.
        """
        return np.load(file_path).astype(np.float32)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["impression_index_batch"],
            batch_data["user_index_batch"],
            batch_data["clicked_news_batch"],
            batch_data["candidate_news_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder(obj): the news encoder of NRMS. 

        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.doc_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(shape=(1, hparams.doc_size,), dtype="int32")
        pred_title_one_reshape = layers.Reshape((hparams.doc_size,))(
            pred_input_title_one
        )

        imp_indexes = keras.Input(shape=(1,), dtype="int32")
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = layers.Embedding(
            hparams.word_size,
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        userencoder = self._build_userencoder(titleencoder)
        newsencoder = titleencoder

        user_present = userencoder(his_input_title)
        news_present = layers.TimeDistributed(newsencoder)(pred_input_title)
        news_present_one = newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [imp_indexes, user_indexes, his_input_title, pred_input_title], preds
        )
        scorer = keras.Model(
            [imp_indexes, user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer
