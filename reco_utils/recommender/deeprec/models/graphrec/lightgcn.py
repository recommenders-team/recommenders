# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import time
import os
import sys
import numpy as np
import pandas as pd
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from reco_utils.common.python_utils import get_top_k_scored_items


class LightGCN(object):
    def __init__(self, hparams, data, seed=None):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function.
        
        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            data (obj): A reco_utils.recommender.deeprec.DataModel.ImplicitCF object, load and process data.
            seed (int): Seed.

        """

        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.data = data
        self.epochs = hparams.epochs
        self.lr = hparams.learning_rate
        self.emb_dim = hparams.embed_size
        self.batch_size = hparams.batch_size
        self.n_layers = hparams.n_layers
        self.decay = hparams.decay
        self.eval_epoch = hparams.eval_epoch
        self.top_k = hparams.top_k
        self.save_model = hparams.save_model
        self.save_epoch = hparams.save_epoch
        self.metrics = hparams.metrics
        self.model_dir = hparams.MODEL_DIR

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
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(
            self.ia_embeddings, self.pos_items
        )
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(
            self.ia_embeddings, self.neg_items
        )
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights["user_embedding"], self.users
        )
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights["item_embedding"], self.pos_items
        )
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights["item_embedding"], self.neg_items
        )

        self.batch_ratings = tf.matmul(
            self.u_g_embeddings,
            self.pos_i_g_embeddings,
            transpose_a=False,
            transpose_b=True,
        )

        self.mf_loss, self.emb_loss = self._create_bpr_loss(
            self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings
        )
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        """Initialize user and item embeddings.

        Returns:
            dict: With keys "user_embedding" and "item_embedding", embeddings of all users and items.

        """
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights["user_embedding"] = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name="user_embedding"
        )
        all_weights["item_embedding"] = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name="item_embedding"
        )
        print("Using xavier initialization.")

        return all_weights

    def _create_lightgcn_embed(self):
        """Calculate the average embeddings of users and items after every layer of the model.

        Returns:
            tf.tensor: average user embeddings
            tf.tensor: average item embeddings

        """
        A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.weights["user_embedding"], self.weights["item_embedding"]], axis=0
        )
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], 0
        )
        return u_g_embeddings, i_g_embeddings

    def _create_bpr_loss(self, users, pos_items, neg_items):
        """Calculate BPR loss.

        Returns:
            tf.Tensor: Matrix factorization loss.
            tf.Tensor: Embedding regularization loss.

        """
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(self.u_g_embeddings_pre)
            + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)
            + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        )
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert a scipy sparse matrix to tf.SparseTensor.

        Returns:
            tf.SparseTensor: SparseTensor after conversion.
            
        """
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def fit(self):
        """Fit the model on self.data.train. If eval_epoch is not -1, evaluate the model on self.data.test
            every "eval_epoch" epoch to observe the training status.

        """
        for epoch in range(1, self.epochs + 1):
            train_start = time.time()
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.opt, self.loss, self.mf_loss, self.emb_loss],
                    feed_dict={
                        self.users: users,
                        self.pos_items: pos_items,
                        self.neg_items: neg_items,
                    },
                )
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            if np.isnan(loss) == True:
                print("ERROR: loss is nan.")
                sys.exit()
            train_end = time.time()
            train_time = train_end - train_start

            if self.save_model and epoch % self.save_epoch == 0:
                save_path_str = os.path.join(self.model_dir, "epoch_" + str(epoch))
                checkpoint_path = self.saver.save(
                    sess=self.sess, save_path=save_path_str
                )

            if self.eval_epoch == -1 or epoch % self.eval_epoch != 0:
                print(
                    "Epoch %d (train)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f"
                    % (epoch, train_time, loss, mf_loss, emb_loss)
                )
            else:
                eval_start = time.time()
                ret = self.run_eval()
                eval_end = time.time()
                eval_time = eval_end - eval_start

                print(
                    "Epoch %d (train)%.1fs + (eval)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f, %s"
                    % (
                        epoch,
                        train_time,
                        eval_time,
                        loss,
                        mf_loss,
                        emb_loss,
                        ", ".join(
                            metric + " = %.5f" % (r)
                            for metric, r in zip(self.metrics, ret)
                        ),
                    )
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
            raise IOError(
                "Failed to find any matching files for {0}".format(model_path)
            )

    def run_eval(self):
        """Run evaluation on self.data.test.

        Returns:
            dict: Results of all metrics in self.metrics.
        """
        topk_scores = self.recommend_k_items(
            self.data.test, top_k=self.top_k, use_id=True
        )
        ret = []
        for metric in self.metrics:
            if metric == "map":
                ret.append(
                    map_at_k(
                        self.data.test, topk_scores, relevancy_method=None, k=self.top_k
                    )
                )
            elif metric == "ndcg":
                ret.append(
                    ndcg_at_k(
                        self.data.test, topk_scores, relevancy_method=None, k=self.top_k
                    )
                )
            elif metric == "precision":
                ret.append(
                    precision_at_k(
                        self.data.test, topk_scores, relevancy_method=None, k=self.top_k
                    )
                )
            elif metric == "recall":
                ret.append(
                    recall_at_k(
                        self.data.test, topk_scores, relevancy_method=None, k=self.top_k
                    )
                )
        return ret

    def score(self, user_ids, remove_seen=True):
        """Score all items for test users.

        Args:
            user_ids (np.array): Users to test.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            np.ndarray: Value of interest of all items for the users.

        """
        if any(np.isnan(user_ids)):
            raise ValueError(
                "LightGCN cannot score users that are not in the training set"
            )
        u_batch_size = self.batch_size
        n_user_batchs = len(user_ids) // u_batch_size + 1
        test_scores = []
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start:end]
            item_batch = range(self.data.n_items)
            rate_batch = self.sess.run(
                self.batch_ratings, {self.users: user_batch, self.pos_items: item_batch}
            )
            test_scores.append(np.array(rate_batch))
        test_scores = np.concatenate(test_scores, axis=0)
        if remove_seen:
            test_scores += self.data.R.tocsr()[user_ids, :] * -np.inf
        return test_scores

    def recommend_k_items(
        self, test, top_k=10, sort_top_k=True, remove_seen=True, use_id=False
    ):
        """Recommend top K items for all users in the test set.

        Args:
            test (pd.DataFrame): Test data.
            top_k (int): Number of top items to recommend.
            sort_top_k (bool): Flag to sort top k results.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            pd.DataFrame: Top k recommendation items for each user.

        """
        data = self.data
        if use_id == False:
            user_ids = np.array([data.user2id[x] for x in test[data.col_user].unique()])
        else:
            user_ids = np.array(test[data.col_user].unique())

        test_scores = self.score(user_ids, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                data.col_user: np.repeat(
                    test[data.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                data.col_item: top_items.flatten()
                if use_id
                else [data.id2item[item] for item in top_items.flatten()],
                data.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()

    def infer_embedding(self, user_file, item_file):
        """Export user and item embeddings to csv files.

        Args:
            user_file (str): Path of file to save user embeddings.
            item_file (str): Path of file to save item embeddings.

        """
        # create output directories if they do not exist
        dirs, _ = os.path.split(user_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        dirs, _ = os.path.split(item_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        data = self.data

        df = pd.DataFrame(
            {
                data.col_user: [data.id2user[id] for id in range(self.n_users)],
                "embedding": list(self.ua_embeddings.eval(session=self.sess)),
            }
        )
        df.to_csv(user_file, sep=" ", index=False)

        df = pd.DataFrame(
            {
                data.col_item: [data.id2item[id] for id in range(self.n_items)],
                "embedding": list(self.ia_embeddings.eval(session=self.sess)),
            }
        )
        df.to_csv(item_file, sep=" ", index=False)
