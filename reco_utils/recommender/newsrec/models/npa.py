# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import PersonalizedAttentivePooling

__all__ = ["NPAModel"]


class NPAModel(BaseModel):
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie:
    NPA: Neural News Recommendation with Personalized Attention, KDD 2019, ADS track.

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator):
        """Initialization steps for MANL.
        Compared with the BaseModel, NPA need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train(obj): NPA data loader class for train data.
            iterator_creator_test(obj): NPA data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator)

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
        """Build NPA model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_npa()
        return model, scorer

    def _build_userencoder(self, titleencoder, user_embedding_layer):
        """The main function to create user encoder of NPA.

        Args:
            titleencoder(obj): the news encoder of NPA. 

        Return:
            obj: the user encoder of NPA.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_id = layers.Reshape((1, 1))(user_indexes)
        repeat_uids = layers.Concatenate(axis=-2)([nuser_id] * hparams.his_size)
        his_title_uid = layers.Concatenate(axis=-1)([his_input_title, repeat_uids])

        click_title_presents = layers.TimeDistributed(titleencoder)(his_title_uid)

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_indexes)
        )
        user_present = PersonalizedAttentivePooling(
            hparams.his_size, hparams.filter_num, hparams.attention_hidden_dim
        )([click_title_presents, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer, user_embedding_layer):
        """The main function to create news encoder of NPA.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NPA.
        """
        hparams = self.hparams
        sequence_title_uindex = keras.Input(
            shape=(hparams.doc_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.doc_size])(
            sequence_title_uindex
        )
        user_index = layers.Lambda(lambda x: x[:, hparams.doc_size :])(
            sequence_title_uindex
        )

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
        )(y)
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = PersonalizedAttentivePooling(
            hparams.doc_size, hparams.filter_num, hparams.attention_hidden_dim
        )([y, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

    def _build_npa(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(shape=(1, hparams.doc_size,), dtype="int32")
        pred_title_one_reshape = layers.Reshape((hparams.doc_size,))(pred_input_title_one)
        imp_indexes = keras.Input(shape=(1,), dtype="int32")
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_index = layers.Reshape((1, 1))(user_indexes)
        repeat_uindex = layers.Concatenate(axis=-2)(
            [nuser_index] * (hparams.npratio + 1)
        )
        pred_title_uindex = layers.Concatenate(axis=-1)(
            [pred_input_title, repeat_uindex]
        )
        pred_title_uindex_one = layers.Concatenate()(
            [pred_title_one_reshape, user_indexes]
        )

        embedding_layer = layers.Embedding(
            hparams.word_size,
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        user_embedding_layer = layers.Embedding(
            hparams.user_num,
            hparams.user_emb_dim,
            trainable=True,
            embeddings_initializer="zeros",
        )

        titleencoder = self._build_newsencoder(embedding_layer, user_embedding_layer)
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

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
