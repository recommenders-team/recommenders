# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf

from recommenders.models.deeprec.models.base_model import BaseModel


__all__ = ["XDeepFMModel"]


class XDeepFMModel(BaseModel):
    """xDeepFM model

    :Citation:

        J. Lian, X. Zhou, F. Zhang, Z. Chen, X. Xie, G. Sun, "xDeepFM: Combining Explicit
        and Implicit Feature Interactions for Recommender Systems", in Proceedings of the
        24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining,
        KDD 2018, London, 2018.
    """

    def _build_graph(self):
        """The main function to create xdeepfm's logic.

        Returns:
            object: The prediction score made by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        with tf.compat.v1.variable_scope("XDeepFM") as scope:  # noqa: F841
            with tf.compat.v1.variable_scope(
                "embedding", initializer=self.initializer
            ) as escope:  # noqa: F841
                self.embedding = tf.compat.v1.get_variable(
                    name="embedding_layer",
                    shape=[hparams.FEATURE_COUNT, hparams.dim],
                    dtype=tf.float32,
                )
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding()

            logit = 0

            if hparams.use_Linear_part:
                print("Add linear part.")
                logit = logit + self._build_linear()

            if hparams.use_FM_part:
                print("Add FM part.")
                logit = logit + self._build_fm()

            if hparams.use_CIN_part:
                print("Add CIN part.")
                if hparams.fast_CIN_d <= 0:
                    logit = logit + self._build_CIN(
                        embed_out, res=True, direct=False, bias=False, is_masked=True
                    )
                else:
                    logit = logit + self._build_fast_CIN(
                        embed_out, res=True, direct=False, bias=False
                    )

            if hparams.use_DNN_part:
                print("Add DNN part.")
                logit = logit + self._build_dnn(embed_out, embed_layer_size)

            return logit

    def _build_embedding(self):
        """The field embedding layer. MLP requires fixed-length vectors as input.
        This function makes sum pooling of feature embeddings for each field.

        Returns:
            embedding:  The result of field embedding layer, with size of #_fields * #_dim.
            embedding_size: #_fields * #_dim
        """
        hparams = self.hparams
        fm_sparse_index = tf.SparseTensor(
            self.iterator.dnn_feat_indices,
            self.iterator.dnn_feat_values,
            self.iterator.dnn_feat_shape,
        )
        fm_sparse_weight = tf.SparseTensor(
            self.iterator.dnn_feat_indices,
            self.iterator.dnn_feat_weights,
            self.iterator.dnn_feat_shape,
        )
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(
            params=self.embedding,
            sp_ids=fm_sparse_index,
            sp_weights=fm_sparse_weight,
            combiner="sum",
        )
        embedding = tf.reshape(
            w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT]
        )
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_linear(self):
        """Construct the linear part for the model.
        This is a linear regression.

        Returns:
            object: Prediction score made by linear regression.
        """
        with tf.compat.v1.variable_scope(
            "linear_part", initializer=self.initializer
        ) as scope:  # noqa: F841
            w = tf.compat.v1.get_variable(
                name="w", shape=[self.hparams.FEATURE_COUNT, 1], dtype=tf.float32
            )
            b = tf.compat.v1.get_variable(
                name="b",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.compat.v1.zeros_initializer(),
            )
            x = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                self.iterator.fm_feat_values,
                self.iterator.fm_feat_shape,
            )
            linear_output = tf.add(tf.sparse.sparse_dense_matmul(x, w), b)
            self.layer_params.append(w)
            self.layer_params.append(b)
            tf.compat.v1.summary.histogram("linear_part/w", w)
            tf.compat.v1.summary.histogram("linear_part/b", b)
            return linear_output

    def _build_fm(self):
        """Construct the factorization machine part for the model.
        This is a traditional 2-order FM module.

        Returns:
            object: Prediction score made by factorization machine.
        """
        with tf.compat.v1.variable_scope("fm_part") as scope:  # noqa: F841
            x = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                self.iterator.fm_feat_values,
                self.iterator.fm_feat_shape,
            )
            xx = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                tf.pow(self.iterator.fm_feat_values, 2),
                self.iterator.fm_feat_shape,
            )
            fm_output = 0.5 * tf.reduce_sum(
                input_tensor=tf.pow(tf.sparse.sparse_dense_matmul(x, self.embedding), 2)
                - tf.sparse.sparse_dense_matmul(xx, tf.pow(self.embedding, 2)),
                axis=1,
                keepdims=True,
            )
            return fm_output

    def _build_CIN(
        self, nn_input, res=False, direct=False, bias=False, is_masked=False
    ):
        """Construct the compressed interaction network.
        This component provides explicit and vector-wise higher-order feature interactions.

        Args:
            nn_input (object): The output of field-embedding layer. This is the input for CIN.
            res (bool): Whether use residual structure to fuse the results from each layer of CIN.
            direct (bool): If true, then all hidden units are connected to both next layer and output layer;
                    otherwise, half of hidden units are connected to next layer and the other half will be connected to output layer.
            bias (bool): Whether to add bias term when calculating the feature maps.
            is_masked (bool): Controls whether to remove self-interaction in the first layer of CIN.

        Returns:
            object: Prediction score made by CIN.
        """
        hparams = self.hparams
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.compat.v1.variable_scope(
            "exfm_part", initializer=self.initializer
        ) as scope:  # noqa: F841
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(
                    split_tensor0, split_tensor, transpose_b=True
                )  # shape :  (Dim, Batch, FieldNum, HiddenNum), a.k.a (D,B,F,H)
                dot_result_o = tf.reshape(
                    dot_result_m,
                    shape=[hparams.dim, -1, field_nums[0] * field_nums[-1]],
                )  # shape: (D,B,FH)
                dot_result = tf.transpose(a=dot_result_o, perm=[1, 0, 2])  # (B,D,FH)

                filters = tf.compat.v1.get_variable(
                    name="f_" + str(idx),
                    shape=[1, field_nums[-1] * field_nums[0], layer_size],
                    dtype=tf.float32,
                )

                if is_masked and idx == 0:
                    ones = tf.ones([field_nums[0], field_nums[0]], dtype=tf.float32)
                    mask_matrix = tf.linalg.band_part(
                        ones, 0, -1
                    ) - tf.linalg.tensor_diag(tf.ones(field_nums[0]))
                    mask_matrix = tf.reshape(
                        mask_matrix, shape=[1, field_nums[0] * field_nums[0]]
                    )

                    dot_result = tf.multiply(dot_result, mask_matrix) * 2
                    self.dot_result = dot_result

                curr_out = tf.nn.conv1d(
                    input=dot_result, filters=filters, stride=1, padding="VALID"
                )  # shape : (B,D,H`)

                if bias:
                    b = tf.compat.v1.get_variable(
                        name="f_b" + str(idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.compat.v1.zeros_initializer(),
                    )
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)

                if hparams.enable_BN is True:
                    curr_out = tf.compat.v1.layers.batch_normalization(
                        curr_out,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                curr_out = self._activate(curr_out, hparams.cross_activation)

                curr_out = tf.transpose(a=curr_out, perm=[0, 2, 1])  # shape : (B,H,D)

                if direct:
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(
                            curr_out, 2 * [int(layer_size / 2)], 1
                        )
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)

            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(input_tensor=result, axis=-1)  # shape : (B,H)

            if res:
                base_score = tf.reduce_sum(
                    input_tensor=result, axis=1, keepdims=True
                )  # (B,1)
            else:
                base_score = 0

            w_nn_output = tf.compat.v1.get_variable(
                name="w_nn_output", shape=[final_len, 1], dtype=tf.float32
            )
            b_nn_output = tf.compat.v1.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.compat.v1.zeros_initializer(),
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = base_score + tf.compat.v1.nn.xw_plus_b(
                result, w_nn_output, b_nn_output
            )
            return exFM_out

    def _build_fast_CIN(self, nn_input, res=False, direct=False, bias=False):
        """Construct the compressed interaction network with reduced parameters.
        This component provides explicit and vector-wise higher-order feature interactions.
        Parameters from the filters are reduced via a matrix decomposition method.
        Fast CIN is more space and time efficient than CIN.

        Args:
            nn_input (object): The output of field-embedding layer. This is the input for CIN.
            res (bool): Whether use residual structure to fuse the results from each layer of CIN.
            direct (bool): If true, then all hidden units are connected to both next layer and output layer;
                    otherwise, half of hidden units are connected to next layer and the other half will be connected to output layer.
            bias (bool): Whether to add bias term when calculating the feature maps.

        Returns:
            object: Prediction score made by fast CIN.
        """
        hparams = self.hparams
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        fast_CIN_d = hparams.fast_CIN_d
        nn_input = tf.reshape(
            nn_input, shape=[-1, int(field_num), hparams.dim]
        )  # (B,F,D)
        nn_input = tf.transpose(a=nn_input, perm=[0, 2, 1])  # (B,D,F)
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        with tf.compat.v1.variable_scope(
            "exfm_part", initializer=self.initializer
        ) as scope:  # noqa: F841
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                if idx == 0:
                    fast_w = tf.compat.v1.get_variable(
                        "fast_CIN_w_" + str(idx),
                        shape=[1, field_nums[0], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )

                    self.cross_params.append(fast_w)
                    dot_result_1 = tf.nn.conv1d(
                        input=nn_input, filters=fast_w, stride=1, padding="VALID"
                    )  # shape: (B,D,d*H)
                    dot_result_2 = tf.nn.conv1d(
                        input=tf.pow(nn_input, 2),
                        filters=tf.pow(fast_w, 2),
                        stride=1,
                        padding="VALID",
                    )  # shape: ((B,D,d*H)
                    dot_result = tf.reshape(
                        0.5 * (dot_result_1 - dot_result_2),
                        shape=[-1, hparams.dim, layer_size, fast_CIN_d],
                    )
                    curr_out = tf.reduce_sum(
                        input_tensor=dot_result, axis=3, keepdims=False
                    )  # shape: ((B,D,H)
                else:
                    fast_w = tf.compat.v1.get_variable(
                        "fast_CIN_w_" + str(idx),
                        shape=[1, field_nums[0], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )
                    fast_v = tf.compat.v1.get_variable(
                        "fast_CIN_v_" + str(idx),
                        shape=[1, field_nums[-1], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )

                    self.cross_params.append(fast_w)
                    self.cross_params.append(fast_v)

                    dot_result_1 = tf.nn.conv1d(
                        input=nn_input, filters=fast_w, stride=1, padding="VALID"
                    )  # shape: ((B,D,d*H)
                    dot_result_2 = tf.nn.conv1d(
                        input=hidden_nn_layers[-1],
                        filters=fast_v,
                        stride=1,
                        padding="VALID",
                    )  # shape: ((B,D,d*H)
                    dot_result = tf.reshape(
                        tf.multiply(dot_result_1, dot_result_2),
                        shape=[-1, hparams.dim, layer_size, fast_CIN_d],
                    )
                    curr_out = tf.reduce_sum(
                        input_tensor=dot_result, axis=3, keepdims=False
                    )  # shape: ((B,D,H)

                if bias:
                    b = tf.compat.v1.get_variable(
                        name="f_b" + str(idx),
                        shape=[1, 1, layer_size],
                        dtype=tf.float32,
                        initializer=tf.compat.v1.zeros_initializer(),
                    )
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)

                if hparams.enable_BN is True:
                    curr_out = tf.compat.v1.layers.batch_normalization(
                        curr_out,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                curr_out = self._activate(curr_out, hparams.cross_activation)

                if direct:
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(
                            curr_out, 2 * [int(layer_size / 2)], 2
                        )
                        final_len += int(layer_size / 2)
                        field_nums.append(int(layer_size / 2))
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                        field_nums.append(int(layer_size))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

            result = tf.concat(final_result, axis=2)
            result = tf.reduce_sum(input_tensor=result, axis=1, keepdims=False)  # (B,H)

            if res:
                base_score = tf.reduce_sum(
                    input_tensor=result, axis=1, keepdims=True
                )  # (B,1)
            else:
                base_score = 0

            w_nn_output = tf.compat.v1.get_variable(
                name="w_nn_output", shape=[final_len, 1], dtype=tf.float32
            )
            b_nn_output = tf.compat.v1.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.compat.v1.zeros_initializer(),
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = (
                tf.compat.v1.nn.xw_plus_b(result, w_nn_output, b_nn_output) + base_score
            )

        return exFM_out

    def _build_dnn(self, embed_out, embed_layer_size):
        """Construct the MLP part for the model.
        This components provides implicit higher-order feature interactions.

        Args:
            embed_out (object): The output of field-embedding layer. This is the input for DNN.
            embed_layer_size (object): Shape of the embed_out

        Returns:
            object: Prediction score made by fast CIN.
        """
        hparams = self.hparams
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.compat.v1.variable_scope(
            "nn_part", initializer=self.initializer
        ) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.compat.v1.get_variable(
                    name="w_nn_layer" + str(layer_idx),
                    shape=[last_layer_size, layer_size],
                    dtype=tf.float32,
                )
                curr_b_nn_layer = tf.compat.v1.get_variable(
                    name="b_nn_layer" + str(layer_idx),
                    shape=[layer_size],
                    dtype=tf.float32,
                    initializer=tf.compat.v1.zeros_initializer(),
                )
                tf.compat.v1.summary.histogram(
                    "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                )
                tf.compat.v1.summary.histogram(
                    "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                )
                curr_hidden_nn_layer = tf.compat.v1.nn.xw_plus_b(
                    hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer
                )
                scope = "nn_part" + str(idx)  # noqa: F841
                activation = hparams.activation[idx]

                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.compat.v1.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                curr_hidden_nn_layer = self._active_layer(
                    logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                )
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.compat.v1.get_variable(
                name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
            )
            b_nn_output = tf.compat.v1.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.compat.v1.zeros_initializer(),
            )
            tf.compat.v1.summary.histogram(
                "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
            )
            tf.compat.v1.summary.histogram(
                "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.compat.v1.nn.xw_plus_b(
                hidden_nn_layers[-1], w_nn_output, b_nn_output
            )
            return nn_output
