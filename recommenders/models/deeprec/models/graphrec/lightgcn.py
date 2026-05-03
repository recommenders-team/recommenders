# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.utils.python_utils import get_top_k_scored_items

logger = logging.getLogger(__name__)
MODEL_CHECKPOINT = "model.pt"


class LightGCN(nn.Module):
    """LightGCN model

    :Citation:

        He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
        "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
        preprint arXiv:2002.02126, 2020.
    """

    def __init__(
        self,
        hparams: Any,
        data: ImplicitCF,
        seed: int | None = None,
    ) -> None:
        """Initializing the model. Create parameters, embeddings, and graph buffers.

        Args:
            hparams (HParams): A HParams object, hold the entire set of hyperparameters.
            data (object): A recommenders.models.deeprec.DataModel.ImplicitCF object, load and process data.
            seed (int): Seed.

        """

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        self.seed = seed

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

        # Trainable embeddings (matches TF VarianceScaling fan_avg uniform == xavier_uniform)
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        logger.info("Using xavier initialization.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.to(self.device)

        self.optimizer = None

    @property
    def ua_embeddings(self) -> torch.Tensor:
        """Aggregated (LGC-propagated) user embeddings."""
        u_g, _ = self._propagate()
        return u_g

    @property
    def ia_embeddings(self) -> torch.Tensor:
        """Aggregated (LGC-propagated) item embeddings."""
        _, i_g = self._propagate()
        return i_g

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LightGCN propagation and return averaged user/item embeddings.

        Uses an iterative sum accumulator (mathematically equivalent to
        ``mean(stack([E^0, ..., E^K]))``) which avoids materializing the
        ``stack`` tensor and shaves one kernel launch per call.
        """
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        sum_embeddings = ego_embeddings
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.A_hat, ego_embeddings)
            sum_embeddings = sum_embeddings + ego_embeddings

        avg_embeddings = sum_embeddings / (self.n_layers + 1)
        u_g, i_g = torch.split(avg_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g, i_g

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass returning embeddings used for BPR loss.

        Args:
            users (torch.LongTensor): User indices.
            pos_items (torch.LongTensor): Positive item indices.
            neg_items (torch.LongTensor): Negative item indices.

        Returns:
            tuple: Propagated and pre-propagation embeddings for users, pos and neg items.
        """
        u_g, i_g = self._propagate()
        u_emb = u_g[users]
        pos_emb = i_g[pos_items]
        neg_emb = i_g[neg_items]
        u_pre = self.user_embedding(users)
        pos_pre = self.item_embedding(pos_items)
        neg_pre = self.item_embedding(neg_items)
        return u_emb, pos_emb, neg_emb, u_pre, pos_pre, neg_pre

    def _bpr_loss(
        self,
        u_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        u_pre: torch.Tensor,
        pos_pre: torch.Tensor,
        neg_pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate BPR loss.

        Returns:
            tuple: Matrix factorization loss and embedding regularization loss.
        """
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        # tf.nn.l2_loss(x) == 0.5 * sum(x ** 2)
        regularizer = 0.5 * (
            u_pre.pow(2).sum() + pos_pre.pow(2).sum() + neg_pre.pow(2).sum()
        )
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X: sp.spmatrix) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse_coo_tensor.

        Returns:
            torch.Tensor: Sparse COO tensor.
        """
        coo = X.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def fit(self) -> None:
        """Fit the model on `self.data.train`. If `eval_epoch` is not -1, evaluate the model on
        `self.data.test` every `eval_epoch` epoch to observe the training status.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            train_start = time.time()
            self.train()

            # Accumulate losses on the device to avoid CPU<->GPU sync per batch
            # (TF v1 sess.run returns all values in one C++ call; the closest
            # PyTorch analogue is to defer .item() until end-of-epoch).
            loss_acc = torch.zeros((), device=self.device)
            mf_acc = torch.zeros((), device=self.device)
            emb_acc = torch.zeros((), device=self.device)

            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for _ in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                users_t = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(
                    self.device, non_blocking=True
                )
                pos_t = torch.from_numpy(np.asarray(pos_items, dtype=np.int64)).to(
                    self.device, non_blocking=True
                )
                neg_t = torch.from_numpy(np.asarray(neg_items, dtype=np.int64)).to(
                    self.device, non_blocking=True
                )

                self.optimizer.zero_grad(set_to_none=True)
                u_emb, pos_emb, neg_emb, u_pre, pos_pre, neg_pre = self.forward(
                    users_t, pos_t, neg_t
                )
                batch_mf_loss, batch_emb_loss = self._bpr_loss(
                    u_emb, pos_emb, neg_emb, u_pre, pos_pre, neg_pre
                )
                batch_loss = batch_mf_loss + batch_emb_loss

                batch_loss.backward()
                self.optimizer.step()

                loss_acc += batch_loss.detach()
                mf_acc += batch_mf_loss.detach()
                emb_acc += batch_emb_loss.detach()

            # Single CPU sync per epoch
            loss = (loss_acc / n_batch).item()
            mf_loss = (mf_acc / n_batch).item()
            emb_loss = (emb_acc / n_batch).item()

            if np.isnan(loss):
                logger.error("loss is nan.")
                sys.exit()

            train_time = time.time() - train_start

            if self.save_model and epoch % self.save_epoch == 0:
                save_path_str = os.path.join(self.model_dir, "epoch_" + str(epoch))
                if not os.path.exists(save_path_str):
                    os.makedirs(save_path_str)
                torch.save(
                    self.state_dict(),
                    os.path.join(save_path_str, MODEL_CHECKPOINT),
                )
                logger.info(
                    "Save model to path %s", os.path.abspath(save_path_str)
                )

            if self.eval_epoch == -1 or epoch % self.eval_epoch != 0:
                logger.info(
                    "Epoch %d (train)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f",
                    epoch,
                    train_time,
                    loss,
                    mf_loss,
                    emb_loss,
                )
            else:
                eval_start = time.time()
                ret = self.run_eval()
                eval_time = time.time() - eval_start

                logger.info(
                    "Epoch %d (train)%.1fs + (eval)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f, %s",
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

    def load(self, model_path: str | None = None) -> None:
        """Load an existing model.

        Args:
            model_path (str): Path to a checkpoint file or a directory containing the
                ``model.pt`` checkpoint.

        Raises:
            IOError: if the restore operation failed.
        """
        try:
            if model_path is not None and os.path.isdir(model_path):
                model_path = os.path.join(model_path, MODEL_CHECKPOINT)
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
        except Exception:
            raise IOError(
                "Failed to find any matching files for {0}".format(model_path)
            )

    def run_eval(self) -> list[float]:
        """Run evaluation on `self.data.test`.

        Returns:
            list: Results for all metrics in `self.metrics`.
        """
        topk_scores = self.recommend_k_items(
            self.data.test, top_k=self.top_k, use_id=True
        )
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

    def score(self, user_ids: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        """Score all items for the given users.

        Args:
            user_ids (np.array): Users to test.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            numpy.ndarray: Scores of all items for each user, shape (len(user_ids), n_items).
        """
        if any(np.isnan(user_ids)):
            raise ValueError(
                "LightGCN cannot score users that are not in the training set"
            )

        u_batch_size = self.batch_size
        n_user_batchs = len(user_ids) // u_batch_size + 1

        self.eval()
        test_scores = []
        with torch.no_grad():
            u_g, i_g = self._propagate()
            for u_batch_id in range(n_user_batchs):
                start = u_batch_id * u_batch_size
                end = (u_batch_id + 1) * u_batch_size
                user_batch = user_ids[start:end]
                if len(user_batch) == 0:
                    continue
                user_batch_t = torch.LongTensor(np.asarray(user_batch)).to(self.device)
                rate_batch = u_g[user_batch_t] @ i_g.t()
                test_scores.append(rate_batch.cpu().numpy())

        test_scores = np.concatenate(test_scores, axis=0)
        if remove_seen:
            test_scores += self.data.R.tocsr()[user_ids, :] * -np.inf
        return test_scores

    def recommend_k_items(
        self,
        test: pd.DataFrame,
        top_k: int = 10,
        sort_top_k: bool = True,
        remove_seen: bool = True,
        use_id: bool = False,
    ) -> pd.DataFrame:
        """Recommend top K items for all users in the test set.

        Args:
            test (pandas.DataFrame): Test data.
            top_k (int): Number of top items to recommend.
            sort_top_k (bool): Flag to sort top k results.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            pandas.DataFrame: Top k recommendation items for each user.
        """
        data = self.data
        if not use_id:
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

    def output_embeddings(
        self,
        idmapper: dict[int, Any],
        n: int,
        target: torch.Tensor,
        user_file: str,
    ) -> None:
        embeddings = target.detach().cpu().numpy()
        with open(user_file, "w") as wt:
            for i in range(n):
                wt.write(
                    "{0}\t{1}\n".format(
                        idmapper[i], " ".join([str(a) for a in embeddings[i]])
                    )
                )

    def infer_embedding(self, user_file: str, item_file: str) -> None:
        """Export user and item embeddings to csv files.

        Args:
            user_file (str): Path of file to save user embeddings.
            item_file (str): Path of file to save item embeddings.
        """
        dirs, _ = os.path.split(user_file)
        if dirs and not os.path.exists(dirs):
            os.makedirs(dirs)
        dirs, _ = os.path.split(item_file)
        if dirs and not os.path.exists(dirs):
            os.makedirs(dirs)

        data = self.data

        self.eval()
        with torch.no_grad():
            u_g, i_g = self._propagate()

        self.output_embeddings(data.id2user, self.n_users, u_g, user_file)
        self.output_embeddings(data.id2item, self.n_items, i_g, item_file)
