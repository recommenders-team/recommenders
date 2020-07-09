# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf

from reco_utils.recommender.deeprec.models.base_model import BaseModel

__all__ = ["DKN"]


class DKN(BaseModel):
    """DKN model (Deep Knowledge-Aware Network)

    H. Wang, F. Zhang, X. Xie and M. Guo, "DKN: Deep Knowledge-Aware Network for News 
    Recommendation", in Proceedings of the 2018 World Wide Web Conference on World 
    Wide Web, 2018.
    """

    def __init__(self, hparams, iterator_creator):
        """Initialization steps for DKN.
        Compared with the BaseModel, DKN requires two different pre-computed embeddings,
        i.e. word embedding and entity embedding.
        After creating these two embedding variables, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters.
            iterator_creator (obj): DKN data loader class.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("embedding"):
                self.embedding = tf.Variable(
                    tf.constant(
                        0.0, shape=[hparams.word_size, hparams.dim], dtype=tf.float32
                    ),
                    trainable=True,
                    name="word",
                )

                word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
                self.init_embedding = self.embedding.assign(word2vec_embedding)
                self.entity_embedding = tf.Variable(
                    tf.constant(
                        0.0, shape=[hparams.entity_size, hparams.dim], dtype=tf.float32
                    ),
                    trainable=True,
                    name="entity",
                )

                e_embedding = self._init_embedding(hparams.entityEmb_file)
                W = tf.Variable(
                    tf.random_uniform([hparams.entity_dim, hparams.dim], -1, 1)
                )
                b = tf.Variable(tf.zeros([hparams.dim]))
                e_embedding_transformed = tf.nn.tanh(tf.matmul(e_embedding, W) + b)
                self.entity_embedding.assign(e_embedding_transformed)

                if hparams.use_context:
                    self.context_embedding = tf.Variable(
                        tf.constant(
                            0.0, shape=[hparams.entity_size, hparams.dim], dtype=tf.float32
                        ),
                        trainable=True,
                        name="context",
                    )
                    c_embedding = self._init_embedding(hparams.contextEmb_file)
                    W = tf.Variable(
                        tf.random_uniform([hparams.entity_dim, hparams.dim], -1, 1)
                    )
                    b = tf.Variable(tf.zeros([hparams.dim]))
                    c_embedding_transformed = tf.nn.tanh(tf.matmul(c_embedding, W) + b)
                    self.context_embedding.assign(c_embedding_transformed)

        super().__init__(hparams, iterator_creator, graph=self.graph)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.
        
        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            obj: A constant tensor.
        """
        return tf.constant(np.load(file_path).astype(np.float32))

    def _l2_loss(self):
        hparams = self.hparams
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        l2_loss = tf.add(
            l2_loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.embedding))
        )
        l2_loss = tf.add(
            l2_loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.entity_embedding))
        )
        if hparams.use_context:
            l2_loss = tf.add(
                l2_loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.context_embedding))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        hparams = self.hparams
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        l1_loss = tf.add(
            l1_loss, tf.multiply(hparams.embed_l1, tf.norm(self.embedding, ord=1))
        )
        l1_loss = tf.add(
            l1_loss,
            tf.multiply(hparams.embed_l1, tf.norm(self.entity_embedding, ord=1)),
        )
        if hparams.use_context:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(hparams.embed_l1, tf.norm(self.context_embedding, ord=1)),
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _build_graph(self):
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        with tf.variable_scope("DKN") as scope:
            logit = self._build_dkn()
            return logit

    def _build_dkn(self):
        """The main function to create DKN's logic.
        
        Returns:
            obj: Prediction score made by the DKN model.
        """
        hparams = self.hparams
        # build attention model for clicked news and candidate news
        click_news_embed_batch, candidate_news_embed_batch = self._build_pair_attention(
            self.iterator.click_news_indices,
            self.iterator.click_news_values,
            self.iterator.click_news_shape,
            hparams,
        )

        nn_input = tf.concat(
            [click_news_embed_batch, candidate_news_embed_batch], axis=1
        )

        dnn_channel_part = 2
        last_layer_size = dnn_channel_part * self.num_filters_total
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(
                    name="w_nn_layer" + str(layer_idx),
                    shape=[last_layer_size, layer_size],
                    dtype=tf.float32,
                )
                curr_b_nn_layer = tf.get_variable(
                    name="b_nn_layer" + str(layer_idx),
                    shape=[layer_size],
                    dtype=tf.float32,
                )
                curr_hidden_nn_layer = tf.nn.xw_plus_b(
                    hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer
                )
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(
                    logit=curr_hidden_nn_layer, activation=activation
                )
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(
                name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
            )
            b_nn_output = tf.get_variable(
                name="b_nn_output", shape=[1], dtype=tf.float32
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output

    def _build_pair_attention(self, field_indices, field_values, field_shape, hparams):
        """This function learns the candidate news article's embedding and user embedding.
        User embedding is generated from click history and also depends on the candidate news article via attention mechanism.
        Article embedding is generated via KCNN module.
        Args:
            field_indices (obj): sparse tensor indices for constructing user clicked history
            field_values (obj): sparse tensor values for constructing user clicked history
            field_shape (obj): sparse tensor shape for constructing user clicked history
            hparams (obj): global hyper-parameters

        Returns:
            click_field_embed_final_batch: user embedding
            news_field_embed_final_batch: candidate news article embedding

        """
        doc_size = hparams.doc_size
        attention_hidden_sizes = hparams.attention_layer_sizes

        candidate_word_batch = self.iterator.candidate_news_index_batch
        click_word_batch = tf.SparseTensor(field_indices, field_values, field_shape)
        click_word_split = tf.sparse_split(
            axis=0, num_split=hparams.batch_size, sp_input=click_word_batch
        )
        news_word_split = tf.split(
            axis=0, num_or_size_splits=hparams.batch_size, value=candidate_word_batch
        )

        candidate_entity_batch = self.iterator.candidate_news_entity_index_batch
        news_entity_split = tf.split(
            axis=0, num_or_size_splits=hparams.batch_size, value=candidate_entity_batch
        )

        field_entities = self.iterator.click_news_entity_values
        click_entity_batch = tf.SparseTensor(field_indices, field_entities, field_shape)
        click_entity_split = tf.sparse_split(
            axis=0, num_split=hparams.batch_size, sp_input=click_entity_batch
        )

        click_field_embed_final_batch = []
        news_field_embed_final_batch = []
        self.news_field_embed_final_batch = []

        with tf.variable_scope("kims_cnn") as kcnn_scope:
            pass

        with tf.variable_scope("attention_net", initializer=self.initializer) as scope:
            for index, news_word in enumerate(news_word_split):
                click_word = click_word_split[index]
                # get non-zero val
                click_word = click_word.values
                click_word = tf.reshape(click_word, [-1, doc_size])

                news_entity = news_entity_split[index]
                click_entity = click_entity_split[index]
                click_entity = click_entity.values
                click_entity = tf.reshape(click_entity, [-1, doc_size])

                # use kims cnn to get conv embedding
                with tf.variable_scope(
                    kcnn_scope, initializer=self.initializer
                ) as cnn_scope:
                    if index > 0:
                        cnn_scope.reuse_variables()
                    news_field_embed = self._kims_cnn(news_word, news_entity, hparams)
                    cnn_scope.reuse_variables()
                    click_field_embed = self._kims_cnn(
                        click_word, click_entity, hparams
                    )

                avg_strategy = False
                if avg_strategy:
                    click_field_embed_final = tf.reduce_mean(
                        click_field_embed, axis=0, keepdims=True
                    )
                else:
                    news_field_embed_repeat = tf.add(
                        tf.zeros_like(click_field_embed), news_field_embed
                    )
                    attention_x = tf.concat(
                        axis=1, values=[click_field_embed, news_field_embed_repeat]
                    )
                    attention_w = tf.get_variable(
                        name="attention_hidden_w",
                        shape=[self.num_filters_total * 2, attention_hidden_sizes],
                        dtype=tf.float32,
                    )
                    attention_b = tf.get_variable(
                        name="attention_hidden_b",
                        shape=[attention_hidden_sizes],
                        dtype=tf.float32,
                    )
                    curr_attention_layer = tf.nn.xw_plus_b(
                        attention_x, attention_w, attention_b
                    )

                    activation = hparams.attention_activation
                    curr_attention_layer = self._active_layer(
                        logit=curr_attention_layer, activation=activation
                    )
                    attention_output_w = tf.get_variable(
                        name="attention_output_w",
                        shape=[attention_hidden_sizes, 1],
                        dtype=tf.float32,
                    )
                    attention_output_b = tf.get_variable(
                        name="attention_output_b", shape=[1], dtype=tf.float32
                    )
                    attention_weight = tf.nn.sigmoid(
                        tf.nn.xw_plus_b(
                            curr_attention_layer, attention_output_w, attention_output_b
                        )
                    )
                    # normalization to the weight sum equal to 1
                    weight_sum = tf.reduce_sum(attention_weight)
                    norm_attention_weight = tf.div(attention_weight, weight_sum)
                    click_field_embed_final = tf.reduce_sum(
                        tf.multiply(click_field_embed, norm_attention_weight),
                        axis=0,
                        keepdims=True,
                    )
                    if attention_w not in self.layer_params:
                        self.layer_params.append(attention_w)
                    if attention_b not in self.layer_params:
                        self.layer_params.append(attention_b)
                    if attention_output_w not in self.layer_params:
                        self.layer_params.append(attention_output_w)
                    if attention_output_b not in self.layer_params:
                        self.layer_params.append(attention_output_b)

                self.news_field_embed_final_batch.append(news_field_embed)
                click_field_embed_final_batch.append(click_field_embed_final)
                scope.reuse_variables()

        click_field_embed_final_batch = tf.concat(click_field_embed_final_batch, axis=0)
        self.news_field_embed_final_batch = tf.concat(self.news_field_embed_final_batch, axis=0)

        return click_field_embed_final_batch, self.news_field_embed_final_batch

    def _kims_cnn(self, word, entity, hparams):
        """The KCNN module. KCNN is an extension of traditional CNN that incorporates symbolic knowledge from
        a knowledge graph into sentence representation learning.
        Args:
            word (obj): word indices for the sentence.
            entity (obj): entity indices for the sentence. Entities are aligned with words in the sentence.
            hparams (obj): global hyper-parameters.

        Returns:
            obj: Sentence representation.
        """
        # kims cnn parameter
        filter_sizes = hparams.filter_sizes
        num_filters = hparams.num_filters

        dim = hparams.dim
        embedded_chars = tf.nn.embedding_lookup(self.embedding, word)

        entity_embedded_chars = tf.nn.embedding_lookup(self.entity_embedding, entity)
        if hparams.use_context:
            context_embedded_chars = tf.nn.embedding_lookup(self.context_embedding, entity)
            concat = tf.concat([embedded_chars, entity_embedded_chars, context_embedded_chars], axis=-1)
        else:
            concat = tf.concat([embedded_chars, entity_embedded_chars], axis=-1)
        concat_expanded = tf.expand_dims(concat, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope(
                "conv-maxpool-%s" % filter_size, initializer=self.initializer
            ):
                # Convolution Layer
                if hparams.use_context:
                    filter_shape = [filter_size, dim * 3, 1, num_filters]
                else:
                    filter_shape = [filter_size, dim * 2, 1, num_filters]
                W = tf.get_variable(
                    name="W" + "_filter_size_" + str(filter_size),
                    shape=filter_shape,
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                )
                b = tf.get_variable(
                    name="b" + "_filter_size_" + str(filter_size),
                    shape=[num_filters],
                    dtype=tf.float32,
                )
                if W not in self.layer_params:
                    self.layer_params.append(W)
                if b not in self.layer_params:
                    self.layer_params.append(b)
                conv = tf.nn.conv2d(
                    concat_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, hparams.doc_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        # self.num_filters_total is the kims cnn output dimension
        self.num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat
