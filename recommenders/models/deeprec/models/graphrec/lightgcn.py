# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from __future__ import annotations

import logging
import os
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

MODEL_CHECKPOINT = "model.pt"

METRIC_OPTIONS = ("map", "ndcg", "precision", "recall")
DEFAULT_METRICS = ("recall", "ndcg", "precision", "map")

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """LightGCN model

    :Citation:

        He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
        "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
        preprint arXiv:2002.02126, 2020.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        norm_adj: sp.spmatrix,
        embed_size: int = 64,
        n_layers: int = 3,
        seed: int | None = None,
    ) -> None:
        """Build the LightGCN model.

        Only the architectural arguments live on the constructor; training-time
        hyperparameters belong on :meth:`fit`.

        Args:
            n_users (int): Number of users.
            n_items (int): Number of items.
            norm_adj (scipy.sparse.spmatrix): Normalized user-item adjacency matrix
                ``D^{-1/2} A D^{-1/2}`` of shape ``(n_users + n_items, n_users + n_items)``.
                Typically obtained via ``ImplicitCF.get_norm_adj_mat()``.
            embed_size (int): Dimension of the user/item embedding tables.
            n_layers (int): Number of light graph convolution layers.
            seed (int): Random seed for embedding initialization.
        """

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        self.seed = seed

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = embed_size
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        # Trainable embeddings (matches TF VarianceScaling fan_avg uniform == xavier_uniform)
        self.user_embedding = nn.Embedding(n_users, embed_size)
        self.item_embedding = nn.Embedding(n_items, embed_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        logger.info("Using xavier initialization.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A_hat = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)
        self.to(self.device)

        # Populated by fit(); inference methods (score / recommend_k_items /
        # run_eval / infer_embedding) read these.
        self.data: ImplicitCF | None = None
        self.batch_size: int = 1024
        self.decay: float = 0.0
        self.optimizer: torch.optim.Optimizer | None = None

    @property
    def ua_embeddings(self) -> torch.Tensor:
        """Aggregated (LGC-propagated) user embeddings.

        Each access runs a full K-layer propagation. If you need both
        user and item embeddings, call :meth:`_propagate` directly to avoid
        recomputing.
        """
        with torch.no_grad():
            u_g, _ = self._propagate()
        return u_g

    @property
    def ia_embeddings(self) -> torch.Tensor:
        """Aggregated (LGC-propagated) item embeddings.

        Each access runs a full K-layer propagation. If you need both
        user and item embeddings, call :meth:`_propagate` directly to avoid
        recomputing.
        """
        with torch.no_grad():
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

    def fit(
        self,
        data: ImplicitCF,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        decay: float = 1e-5,
        eval_epoch: int = 5,
        top_k: int = 10,
        metrics: list[str] | None = None,
        save_model: bool = False,
        save_epoch: int = 5,
        model_dir: str = "./",
    ) -> None:
        """Fit the model on ``data.train``.

        Calling ``fit`` multiple times retrains the *same* model (parameters are
        not re-initialized). Inference methods (``score``, ``recommend_k_items``,
        ``run_eval``, ``infer_embedding``) read ``self.data`` set here.

        Args:
            data (ImplicitCF): Training/test container.
            epochs (int): Number of training epochs.
            learning_rate (float): Adam learning rate.
            batch_size (int): Mini-batch size for both training and inference scoring.
            decay (float): L2 regularization coefficient on the input embeddings.
            eval_epoch (int): If positive, run :meth:`run_eval` every ``eval_epoch`` epochs.
                ``-1`` disables periodic evaluation.
            top_k (int): ``k`` used by periodic evaluation.
            metrics (list[str]): Metrics to report during periodic evaluation. Defaults
                to ``["recall", "ndcg", "precision", "map"]``.
            save_model (bool): If True, dump checkpoints under ``model_dir``.
            save_epoch (int): Save a checkpoint every ``save_epoch`` epochs (only used
                when ``save_model`` is True).
            model_dir (str): Directory to write checkpoints to.
        """
        if metrics is None:
            metrics = list(DEFAULT_METRICS)
        _validate_metrics(metrics)

        self.data = data
        self.batch_size = batch_size
        self.decay = decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            train_start = time.time()
            self.train()

            # Accumulate losses on the device to avoid CPU<->GPU sync per batch
            # (TF v1 sess.run returns all values in one C++ call; the closest
            # PyTorch analogue is to defer .item() until end-of-epoch).
            loss_acc = torch.zeros((), device=self.device)
            mf_acc = torch.zeros((), device=self.device)
            emb_acc = torch.zeros((), device=self.device)

            n_batch = data.train.shape[0] // batch_size + 1
            for _ in range(n_batch):
                users, pos_items, neg_items = data.train_loader(batch_size)
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
                raise RuntimeError("Training diverged: loss is NaN.")

            train_time = time.time() - train_start

            if save_model and epoch % save_epoch == 0:
                save_path_str = os.path.join(model_dir, "epoch_" + str(epoch))
                if not os.path.exists(save_path_str):
                    os.makedirs(save_path_str)
                torch.save(
                    self.state_dict(),
                    os.path.join(save_path_str, MODEL_CHECKPOINT),
                )
                logger.info("Save model to path %s", os.path.abspath(save_path_str))

            if eval_epoch == -1 or epoch % eval_epoch != 0:
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
                ret = self.run_eval(top_k=top_k, metrics=metrics)
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
                        metric + " = %.5f" % (r) for metric, r in zip(metrics, ret)
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
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.load_state_dict(state_dict)
        except Exception:
            raise IOError(
                "Failed to find any matching files for {0}".format(model_path)
            )

    def run_eval(
        self,
        top_k: int = 10,
        metrics: list[str] | None = None,
    ) -> list[float]:
        """Run evaluation on ``self.data.test``.

        Args:
            top_k (int): Cut-off ``k`` for the ranking metrics.
            metrics (list[str]): Metrics to compute. Defaults to
                ``["recall", "ndcg", "precision", "map"]``.

        Returns:
            list[float]: Metric values, in the same order as ``metrics``.
        """
        if self.data is None:
            raise RuntimeError(
                "run_eval() requires a dataset. Call fit() first or assign self.data."
            )
        if metrics is None:
            metrics = list(DEFAULT_METRICS)
        _validate_metrics(metrics)

        topk_scores = self.recommend_k_items(self.data.test, top_k=top_k, use_id=True)
        ret = []
        for metric in metrics:
            if metric == "map":
                ret.append(map_at_k(self.data.test, topk_scores, k=top_k))
            elif metric == "ndcg":
                ret.append(ndcg_at_k(self.data.test, topk_scores, k=top_k))
            elif metric == "precision":
                ret.append(precision_at_k(self.data.test, topk_scores, k=top_k))
            elif metric == "recall":
                ret.append(recall_at_k(self.data.test, topk_scores, k=top_k))
        return ret

    def score(self, user_ids: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        """Score all items for the given users.

        Args:
            user_ids (np.array): Users to test.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            numpy.ndarray: Scores of all items for each user, shape (len(user_ids), n_items).
        """
        if self.data is None:
            raise RuntimeError(
                "score() requires a dataset. Call fit() first or assign self.data."
            )
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
        if self.data is None:
            raise RuntimeError(
                "recommend_k_items() requires a dataset. Call fit() first or assign self.data."
            )
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
        if self.data is None:
            raise RuntimeError(
                "infer_embedding() requires a dataset. Call fit() first or assign self.data."
            )
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


def _validate_metrics(metrics: list[str]) -> None:
    for metric in metrics:
        if metric not in METRIC_OPTIONS:
            raise ValueError(
                "Wrong metric(s), please select one of this list: {}".format(
                    list(METRIC_OPTIONS)
                )
            )
