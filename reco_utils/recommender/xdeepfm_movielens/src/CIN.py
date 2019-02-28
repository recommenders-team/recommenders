"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["CINModel"]


class CINModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("CIN") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_linear(hparams)
            # res: use resnet?  direct: without split?  reduce_D: Dimension reduction?  f_dim: dimension of reduce_D
            logit = tf.add(logit, self._build_extreme_FM(hparams, embed_out, res=False, direct=False, bias=False, reduce_D=False, f_dim=2))
            # logit = tf.add(logit, self._build_dnn(hparams, embed_out, embed_layer_size))
            return logit

    def _build_embedding(self, hparams):
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output

    def _build_fm(self, hparams):
        with tf.variable_scope("fm_part") as scope:
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            xx = tf.SparseTensor(self.iterator.fm_feat_indices,
                                 tf.pow(self.iterator.fm_feat_values, 2),
                                 self.iterator.fm_feat_shape)
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x, self.embedding), 2) - \
                tf.sparse_tensor_dense_matmul(xx,
                                              tf.pow(self.embedding, 2)), 1,
                keep_dims=True)
            return fm_output
    """
    def _build_extreme_FM_slow_bad(self, hparams, nn_input):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                dot_results = []
                split_tensor = tf.split(hidden_nn_layers[-1], field_nums[-1]*[1], 1)
                for s in split_tensor:
                    s = tf.tile(s, [1, field_nums[0], 1])
                    dot_results.append(tf.multiply(s, hidden_nn_layers[0]))
                dot_result = tf.concat(dot_results, axis=1)
                filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, len(dot_results)*field_nums[0], layer_size],
                                         dtype=tf.float32)
                dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                if idx != len(hparams.cross_layer_sizes)-1:
                    next_hidden, direct_connect = tf.split(curr_out, 2*[int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden=0
                    final_len += layer_size
                
                ###
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                ###
                
                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)
                field_nums.append(int(layer_size / 2))
                # field_nums.append(int(layer_size))
                self.cross_params.append(filters)
            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)
            ###
            # residual network
            w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                          shape=[final_len, 128],
                                          dtype=tf.float32)
            b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                          shape=[128],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output1)
            self.layer_params.append(b_nn_output1)
            exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
            exFM_out1 = self._active_layer(logit=exFM_out0,
                                                      scope=scope,
                                                      activation="relu",
                                                      layer_idx=0)
            w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                           shape=[128 + final_len, 1],
                                           dtype=tf.float32)
            b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                           shape=[1],
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output2)
            self.layer_params.append(b_nn_output2)
            exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
            exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

            ###
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[final_len, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out
    """

    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                if reduce_D:
                    hparams.logger.info("reduce_D")
                    filters0 = tf.get_variable("f0_" + str(idx),
                                               shape=[1, layer_size, field_nums[0], f_dim],
                                               dtype=tf.float32)
                    filters_ = tf.get_variable("f__" + str(idx),
                                               shape=[1, layer_size, f_dim, field_nums[-1]],
                                               dtype=tf.float32)
                    filters_m = tf.matmul(filters0, filters_)
                    filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                    filters = tf.transpose(filters_o, perm=[0, 2, 1])
                else:
                    filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

                # BIAS ADD
                if bias:
                    hparams.logger.info("bias")
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)

                curr_out = self._activate(curr_out, hparams.cross_activation)

                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                if direct:
                    hparams.logger.info("all direct connect")
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    hparams.logger.info("split connect")
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
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
            result = tf.reduce_sum(result, -1)
            if res:
                hparams.logger.info("residual network")
                w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                               shape=[final_len, 128],
                                               dtype=tf.float32)
                b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                               shape=[128],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output1)
                self.layer_params.append(b_nn_output1)
                exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
                exFM_out1 = self._active_layer(logit=exFM_out0,
                                               scope=scope,
                                               activation="relu",
                                               layer_idx=0)
                w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                               shape=[128 + final_len, 1],
                                               dtype=tf.float32)
                b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                               shape=[1],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output2)
                self.layer_params.append(b_nn_output2)
                exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
                exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

            else:
                hparams.logger.info("no residual network")
                w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output)
                self.layer_params.append(b_nn_output)
                exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out

    def _build_extreme_FM_quick(self, hparams, nn_input):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')


                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])


                hparams.logger.info("split connect")
                if idx != len(hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
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
            result = tf.reduce_sum(result, -1)

            hparams.logger.info("no residual network")
            w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out


    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.dim
        """
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
