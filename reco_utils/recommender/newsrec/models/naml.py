# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2

__all__ = ["NAMLModel"]


class NAMLModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for NAML.
        Compared with the BaseModel, NAML need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train(obj): NAML data loader class for train data.
            iterator_creator_test(obj): NAML data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["impression_index_batch"],
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["clicked_body_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_body_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"]
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.
        
        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            np.array: A constant numpy array.
        """
        return np.load(file_path).astype(np.float32)

    def _build_graph(self):
        """Build NAML model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_naml()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NAML.

        Args:
            newsencoder(obj): the news encoder of NAML. 

        Return:
            obj: the user encoder of NAML.
        """
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 2),
            dtype="int32",
        )

        click_news_presents = layers.TimeDistributed(newsencoder)(
            his_input_title_body_verts
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(click_news_presents)

        model = keras.Model(
            his_input_title_body_verts, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NAML.
        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NAML.
        """
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 2,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_body_verts
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size : hparams.title_size + hparams.body_size]
        )(input_title_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[
                :,
                hparams.title_size
                + hparams.body_size : hparams.title_size
                + hparams.body_size
                + 1,
            ]
        )(input_title_body_verts)
        input_subvert = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1 :]
        )(input_title_body_verts)

        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr, vert_repr, subvert_repr]
        )
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)

        model = keras.Model(input_title_body_verts, news_repr, name="news_encoder")
        return model

    def _build_titleencoder(self, embedding_layer):
        """build title encoder of NAML news encoder.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the title encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed)
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        """build body encoder of NAML news encoder.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the body encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype="int32")
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed)
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model

    def _build_vertencoder(self):
        """build vert encoder of NAML news encoder.

        Return:
            obj: the vert encoder of NAML.
        """
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype="int32")

        vert_embedding = layers.Embedding(
            hparams.vert_num, hparams.vert_emb_dim, trainable=True
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams.filter_num, 
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed)
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_subvertencoder(self):
        """build subvert encoder of NAML news encoder.

        Return:
            obj: the subvert encoder of NAML.
        """
        hparams = self.hparams
        input_subvert = keras.Input(shape=(1,), dtype="int32")

        subvert_embedding = layers.Embedding(
            hparams.subvert_num, hparams.subvert_emb_dim, trainable=True
        )

        subvert_emb = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(
            hparams.filter_num, 
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed)
        )(subvert_emb)
        pred_subvert = layers.Reshape((1, hparams.filter_num))(pred_subvert)

        model = keras.Model(input_subvert, pred_subvert, name="subvert_encoder")
        return model

    def _build_naml(self):
        """The main function to create NAML's logic. The core of NAML
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        his_input_subvert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")

        pred_input_title_one = keras.Input(
            shape=(1, hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(1, hparams.body_size,), dtype="int32")
        pred_input_vert_one = keras.Input(shape=(1, 1), dtype="int32")
        pred_input_subvert_one = keras.Input(shape=(1, 1), dtype="int32")

        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_vert, his_input_subvert]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert]
        )

        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ]
        )
        pred_title_body_verts_one = layers.Reshape((-1,))(pred_title_body_verts_one)

        imp_indexes = keras.Input(shape=(1,), dtype="int32")
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = layers.Embedding(
            hparams.word_size,
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        newsencoder = self._build_newsencoder(embedding_layer)
        userencoder = self._build_userencoder(newsencoder)

        user_present = userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(newsencoder)(pred_title_body_verts)
        news_present_one = newsencoder(pred_title_body_verts_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                imp_indexes,
                user_indexes,
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                imp_indexes,
                user_indexes,
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ],
            pred_one,
        )

        return model, scorer
