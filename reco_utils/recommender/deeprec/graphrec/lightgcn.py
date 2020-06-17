# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import time
import os
import sys
import numpy as np
import pandas as pd
from reco_utils.recommender.deeprec.graphrec.ranking_metrics import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.common.python_utils import get_top_k_scored_items
from reco_utils.common import constants


class LightGCN(object):
    def __init__(
        self,
        data,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        seed=None,
        epoch=1000,
        learning_rate=0.001,
        embed_size=64,
        batch_size=2048,
        layer_size=[64, 64, 64],
        decay=1e-4,
        eval_epoch=20,
        top_k=20,
        save_epoch=100,
        metrics=["recall", "ndcg", "precision", "map"],
        model_dir=None
    ):
        """Constructor
        
        Args:
            data (Dataset): initialized Dataset in ./dataset.py
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            col_prediction (str): Prediction column name.
            seed (int): Seed.
            epoch (int): Number of epochs for training.
            learning rate (float): Learning rate.
            embed_size (int): Embedding dimension for all users and items.
            batch_size (int): Batch size.
            layer_size (int): Output sizes of every layer.
            decay (int): Regularization coefficient.
            eval_epoch (int): If it is None, evaluation metrics will not be calculated; otherwise metrics will be calculated
                on test data every "eval_epoch" epochs.
            top_k (int): Parameter k for ranking metrics like ndcg@k.
            save_epoch (int): If it is None, model will not be saved; otherwise save the latest model every "save_epoch" epochs.
            metrics (int): Evaluation metrics.
            model_dir (str): Directory to save model.
        """

        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.data = data
        self.n_fold = 100
        self.epoch = epoch
        self.lr = learning_rate
        self.emb_dim = embed_size
        self.batch_size = batch_size
        self.weight_size = layer_size
        self.n_layers = len(self.weight_size)
        self.decay = decay
        self.eval_epoch = eval_epoch
        self.top_k = top_k
        self.save_epoch = save_epoch
        self.metrics = metrics
        self.model_dir = model_dir

        metric_options = ["map", "ndcg", "precision", "recall"]
        for metric in self.metrics:
            if metric not in metric_options:
                raise ValueError(
                    "Wrong metric(s), please select one of this list: {}".format(
                        metric_options
                    )
                )

        self.norm_adj = data.get_norm_adj_mat()

        self.n_users = data.n_users
        self.n_items = data.n_items

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["user_embedding"], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.neg_items)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        self.mf_loss, self.emb_loss = self._create_bpr_loss(self.u_g_embeddings,
                                                        self.pos_i_g_embeddings,
                                                        self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights["user_embedding"] = tf.Variable(initializer([self.n_users, self.emb_dim]), name="user_embedding")
        all_weights["item_embedding"] = tf.Variable(initializer([self.n_items, self.emb_dim]), name="item_embedding")
        print("Using xavier initialization.")

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        
        ego_embeddings = tf.concat([self.weights["user_embedding"], self.weights["item_embedding"]], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def fit(self):
        for epoch in range(1, self.epoch + 1):
            train_start = time.time()
            loss, mf_loss, emb_loss = 0., 0., 0.
            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run([self.opt, self.loss, self.mf_loss, self.emb_loss],
                                feed_dict={self.users: users, self.pos_items: pos_items, self.neg_items: neg_items})
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            if np.isnan(loss) == True:
                print("ERROR: loss is nan.")
                sys.exit()
            train_end = time.time()
            train_time = train_end - train_start

            if self.model_dir is not None and self.save_epoch is not None and epoch % self.save_epoch == 0:
                save_path_str = os.path.join(self.model_dir, "epoch_" + str(epoch))
                checkpoint_path = self.saver.save(
                    sess=self.sess, save_path=save_path_str
                )

            if self.data.test is None or self.eval_epoch is None or epoch % self.eval_epoch != 0:
                print(
                    "Epoch %d (train)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f" % (
                    epoch, train_time, loss, mf_loss, emb_loss)
                )
            else:
                eval_start = time.time()
                ret = self.run_eval()
                eval_end = time.time()
                eval_time = eval_end - eval_start

                print(
                    "Epoch %d (train)%.1fs + (eval)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f, %s" % \
                    (epoch, train_time, eval_time, loss, mf_loss, emb_loss, 
                    ", ".join(metric + " = %.5f" % (r) for metric, r in zip(self.metrics, ret)))
                )

    def load(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: Model path.

        Raises:
            IOError: if the restore operation failed.
        """
        try:
            self.saver.restore(self.sess, model_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(model_path))

    def run_eval(self):
        topk_scores = self.recommend_k_items(self.data.test, top_k=self.top_k, use_id=True)
        ret = []
        for metric in self.metrics:
            if metric == "map":
                ret.append(map_at_k(self.data.test, topk_scores, k=self.top_k))
            elif metric == "ndcg":
                ret.append(ndcg_at_k(self.data.test, topk_scores, k=self.top_k))
            elif metric == "precision":
                ret.append(precision_at_k(self.data.test, topk_scores, k=self.top_k))
            elif metric == "recall":
                ret.append(recall_at_k(self.data.test, topk_scores, k=self.top_k))
        return ret

    def score(self, user_ids, remove_seen=True):
        if any(np.isnan(user_ids)):
            raise ValueError("LightGCN cannot score users that are not in the training set")
        u_batch_size = self.batch_size
        n_user_batchs = len(user_ids) // u_batch_size + 1
        test_scores = []
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start: end]
            item_batch = range(self.data.n_items)
            rate_batch = self.sess.run(self.batch_ratings, {self.users: user_batch, self.pos_items: item_batch})
            test_scores.append(np.array(rate_batch))
        test_scores = np.concatenate(test_scores, axis=0)
        if remove_seen:
            test_scores += self.data.R.tocsr()[user_ids, :] * -np.inf
        return test_scores

    def recommend_k_items( 
        self, test, top_k=10, sort_top_k=True, remove_seen=True, use_id=False
    ):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): Test data.
            top_k (int): Number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pd.DataFrame: top k recommendation items for each user
        """
        if use_id == False:
            user_ids = np.array([self.data.user2id[x] for x in test[self.col_user].unique()])
        else:
            user_ids = np.array(test[self.col_user].unique())

        test_scores = self.score(user_ids, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test[self.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: top_items.flatten() if use_id else [self.data.id2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()
