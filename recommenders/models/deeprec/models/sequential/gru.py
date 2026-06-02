# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.deeprec.io.torch.sequential_iterator import (
    SequentialIterator,
)


__all__ = ["GRUModel"]

logger = logging.getLogger(__name__)
MODEL_CHECKPOINT = "model.pt"

DEFAULT_METRICS = ("auc", "logloss")
DEFAULT_PAIRWISE_METRICS = ("mean_mrr", "ndcg@10", "group_auc")

_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": F.elu,
    "identity": lambda x: x,
}


class _FCN(nn.Module):
    """MLP head matching :meth:`BaseModel._fcn_net`.

    Each hidden layer is ``Linear -> [BatchNorm1d] -> activation -> dropout``;
    the final layer is a single ``Linear`` producing the logit.
    """

    def __init__(
        self,
        in_dim: int,
        layer_sizes: Sequence[int],
        activations: Sequence[str],
        dropout: Sequence[float],
        enable_BN: bool,
        user_dropout: bool,
    ) -> None:
        super().__init__()
        if len(activations) < len(layer_sizes):
            raise ValueError(
                "activations must have at least len(layer_sizes) entries"
            )
        if len(dropout) < len(layer_sizes):
            raise ValueError("dropout must have at least len(layer_sizes) entries")

        self.layer_sizes = list(layer_sizes)
        self.activations = [a.lower() for a in activations]
        self.dropouts = list(dropout)
        self.enable_BN = enable_BN
        self.user_dropout = user_dropout

        linears = []
        bns = []
        last = in_dim
        for size in self.layer_sizes:
            linears.append(nn.Linear(last, size))
            bns.append(
                nn.BatchNorm1d(size, momentum=0.05, eps=1e-4)
                if enable_BN
                else nn.Identity()
            )
            last = size
        self.linears = nn.ModuleList(linears)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, (linear, bn) in enumerate(zip(self.linears, self.bns)):
            x = linear(x)
            x = bn(x)
            act = _ACTIVATIONS.get(self.activations[idx])
            if act is None:
                raise ValueError(f"activation not supported: {self.activations[idx]}")
            x = act(x)
            if self.user_dropout and self.dropouts[idx] > 0:
                x = F.dropout(x, p=self.dropouts[idx], training=self.training)
        return self.out(x)


class GRUModel(nn.Module):
    """GRU sequential recommender (PyTorch).

    :Citation:

        Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau,
        Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning Phrase
        Representations using RNN Encoder-Decoder for Statistical Machine
        Translation. arXiv preprint arXiv:1406.1078. 2014.
    """

    def __init__(
        self,
        user_vocab_length: int,
        item_vocab_length: int,
        cate_vocab_length: int,
        user_embedding_dim: int = 16,
        item_embedding_dim: int = 32,
        cate_embedding_dim: int = 8,
        max_seq_length: int = 50,
        hidden_size: int = 40,
        layer_sizes: Sequence[int] = (100, 64),
        activations: Sequence[str] = ("relu", "relu"),
        dropout: Sequence[float] = (0.3, 0.3),
        enable_BN: bool = True,
        user_dropout: bool = True,
        seed: int | None = None,
    ) -> None:
        """Build the GRU model.

        Only architectural arguments live on the constructor; training-time
        hyperparameters (epochs, learning_rate, optimizer, regularization,
        loss, ...) belong on :meth:`fit`.

        Args:
            user_vocab_length (int): Size of the user vocabulary.
            item_vocab_length (int): Size of the item vocabulary.
            cate_vocab_length (int): Size of the category vocabulary.
            user_embedding_dim (int): Dimension of user embeddings.
            item_embedding_dim (int): Dimension of item embeddings.
            cate_embedding_dim (int): Dimension of category embeddings.
            max_seq_length (int): Maximum length of the history sequence.
            hidden_size (int): Hidden size of the GRU.
            layer_sizes (Sequence[int]): Hidden sizes of the MLP head.
            activations (Sequence[str]): Activations for the MLP head (one per
                hidden layer).
            dropout (Sequence[float]): Dropout rates for the MLP head (one per
                hidden layer).
            enable_BN (bool): If True, apply BatchNorm1d after each MLP layer.
            user_dropout (bool): If True, apply dropout in the MLP head.
            seed (int): Random seed for parameter initialization.
        """
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        self.seed = seed

        self.user_vocab_length = user_vocab_length
        self.item_vocab_length = item_vocab_length
        self.cate_vocab_length = cate_vocab_length
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.cate_embedding_dim = cate_embedding_dim
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size

        self.user_embedding = nn.Embedding(user_vocab_length, user_embedding_dim)
        self.item_embedding = nn.Embedding(item_vocab_length, item_embedding_dim)
        self.cate_embedding = nn.Embedding(cate_vocab_length, cate_embedding_dim)
        # Matches TF tnormal initializer (stddev=0.01) which is the default
        # init_method in the YAML configs.
        for emb in (self.user_embedding, self.item_embedding, self.cate_embedding):
            nn.init.trunc_normal_(emb.weight, std=0.01)

        rnn_in = item_embedding_dim + cate_embedding_dim
        # batch_first=True so input shape is (B, T, F) and final state is
        # (1, B, H); the trailing dim matches TF dynamic_rnn output.
        self.gru = nn.GRU(rnn_in, hidden_size, batch_first=True)

        # Model output concatenates the GRU final state, the target item
        # embedding (item + cate), and the user embedding. The TF reference
        # built ``self.user_embedding`` but never consumed it in the GRU
        # variant; wiring it into the FCN input (a per-instance, time-
        # invariant feature) is a small improvement, not a port-time bug.
        self.target_dim = item_embedding_dim + cate_embedding_dim
        self.fcn = _FCN(
            in_dim=hidden_size + self.target_dim + user_embedding_dim,
            layer_sizes=layer_sizes,
            activations=activations,
            dropout=dropout,
            enable_BN=enable_BN,
            user_dropout=user_dropout,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Populated by fit() and consumed by run_eval / predict.
        self.iterator: SequentialIterator | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.embed_l2: float = 0.0
        self.embed_l1: float = 0.0
        self.layer_l2: float = 0.0
        self.layer_l1: float = 0.0
        self.train_num_ngs: int = 1
        self.loss_name: str = "softmax"
        self.best_epoch: int = 0

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        cates: torch.Tensor,
        item_history: torch.Tensor,
        item_cate_history: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-instance logit.

        Args:
            users (LongTensor): ``(B,)`` user ids.
            items (LongTensor): ``(B,)`` target item ids.
            cates (LongTensor): ``(B,)`` target category ids.
            item_history (LongTensor): ``(B, T)`` history item ids.
            item_cate_history (LongTensor): ``(B, T)`` history category ids.
            mask (FloatTensor): ``(B, T)`` 1/0 mask for valid history steps.

        Returns:
            torch.Tensor: ``(B, 1)`` logit.
        """
        item_hist_emb = self.item_embedding(item_history)
        cate_hist_emb = self.cate_embedding(item_cate_history)
        history_emb = torch.cat([item_hist_emb, cate_hist_emb], dim=2)

        # pack_padded_sequence requires lengths on CPU as int64 and cannot
        # accept length 0; clamp to 1 for empty-history rows.
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history_emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, final_state = self.gru(packed)
        final_state = final_state.squeeze(0)

        target_emb = torch.cat(
            [self.item_embedding(items), self.cate_embedding(cates)], dim=1
        )
        user_emb = self.user_embedding(users)
        model_output = torch.cat([final_state, target_emb, user_emb], dim=1)
        return self.fcn(model_output)

    # --- losses & regularization ---------------------------------------------

    def _data_loss(self, logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Match :meth:`BaseModel._compute_data_loss`."""
        if self.loss_name == "cross_entropy_loss":
            return F.binary_cross_entropy_with_logits(
                logit.reshape(-1), labels.reshape(-1)
            )
        if self.loss_name == "square_loss":
            pred = torch.sigmoid(logit)
            return torch.sqrt(F.mse_loss(pred.reshape(-1), labels.reshape(-1)))
        if self.loss_name == "log_loss":
            pred = torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7)
            return F.binary_cross_entropy(pred.reshape(-1), labels.reshape(-1))
        if self.loss_name == "softmax":
            group = self.train_num_ngs + 1
            logits = logit.reshape(-1, group)
            labels_g = labels.reshape(-1, group)
            softmax = F.softmax(logits, dim=-1)
            # Equivalent to TF: keep softmax on positives, set padding to 1
            # so log(.) == 0 and the term drops out.
            pos = torch.where(labels_g > 0.5, softmax, torch.ones_like(softmax))
            return -group * torch.mean(torch.log(pos))
        raise ValueError(f"this loss not defined {self.loss_name}")

    def _regular_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        embed_params = [
            self.user_embedding.weight,
            self.item_embedding.weight,
            self.cate_embedding.weight,
        ]
        for p in embed_params:
            # tf.nn.l2_loss(x) == 0.5 * sum(x**2)
            loss = loss + self.embed_l2 * 0.5 * p.pow(2).sum()
            loss = loss + self.embed_l1 * p.abs().sum()

        layer_params: Iterable[torch.nn.Parameter] = list(self.gru.parameters())
        for module in (*self.fcn.linears, self.fcn.out):
            layer_params = (*layer_params, *module.parameters())
        for p in layer_params:
            loss = loss + self.layer_l2 * 0.5 * p.pow(2).sum()
            loss = loss + self.layer_l1 * p.abs().sum()
        return loss

    # --- batch helpers --------------------------------------------------------

    def _to_device(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "labels": torch.from_numpy(batch["labels"]).float().to(self.device),
            "users": torch.from_numpy(batch["users"].astype(np.int64)).to(self.device),
            "items": torch.from_numpy(batch["items"].astype(np.int64)).to(self.device),
            "cates": torch.from_numpy(batch["cates"].astype(np.int64)).to(self.device),
            "item_history": torch.from_numpy(
                batch["item_history"].astype(np.int64)
            ).to(self.device),
            "item_cate_history": torch.from_numpy(
                batch["item_cate_history"].astype(np.int64)
            ).to(self.device),
            "mask": torch.from_numpy(batch["mask"]).float().to(self.device),
        }

    def _build_optimizer(self, name: str, lr: float) -> torch.optim.Optimizer:
        name = name.lower()
        params = self.parameters()
        if name == "adam":
            return torch.optim.Adam(params, lr=lr)
        if name in ("sgd", "gd"):
            return torch.optim.SGD(params, lr=lr)
        if name == "adadelta":
            return torch.optim.Adadelta(params, lr=lr)
        if name == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr)
        raise ValueError(f"optimizer not supported: {name}")

    # --- training -------------------------------------------------------------

    def fit(
        self,
        train_file: str,
        valid_file: str,
        user_vocab: str,
        item_vocab: str,
        cate_vocab: str,
        valid_num_ngs: int,
        train_num_ngs: int = 4,
        epochs: int = 50,
        batch_size: int = 400,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        loss: str = "softmax",
        embed_l2: float = 1e-4,
        embed_l1: float = 0.0,
        layer_l2: float = 1e-4,
        layer_l1: float = 0.0,
        max_grad_norm: float | None = None,
        min_seq_length: int = 1,
        early_stop: int = 10,
        eval_metric: str = "group_auc",
        metrics: Sequence[str] = DEFAULT_METRICS,
        pairwise_metrics: Sequence[str] = DEFAULT_PAIRWISE_METRICS,
        show_step: int = 100,
        save_model: bool = False,
        save_epoch: int = 1,
        model_dir: str = "./",
    ) -> "GRUModel":
        """Train on ``train_file``, evaluating on ``valid_file`` each epoch.

        Args:
            train_file (str): Tab-separated training file.
            valid_file (str): Tab-separated validation file.
            user_vocab (str): Path to user vocab pickle.
            item_vocab (str): Path to item vocab pickle.
            cate_vocab (str): Path to category vocab pickle.
            valid_num_ngs (int): Negatives per positive in the validation file
                (used to reshape predictions for group-wise metrics).
            train_num_ngs (int): Negatives per positive sampled in-batch
                during training. Softmax loss is computed over the resulting
                ``(1 + train_num_ngs)`` group.
            epochs (int): Number of training epochs.
            batch_size (int): Number of positive instances per training batch
                (each row expands to ``batch_size * (1 + train_num_ngs)``).
            learning_rate (float): Optimizer learning rate.
            optimizer (str): One of ``adam``, ``sgd``, ``adadelta``, ``adagrad``,
                ``rmsprop``.
            loss (str): One of ``softmax``, ``cross_entropy_loss``,
                ``square_loss``, ``log_loss``.
            embed_l2 / embed_l1: Regularization on embedding tables.
            layer_l2 / layer_l1: Regularization on GRU + MLP layer params.
            max_grad_norm (float): If given, clip global gradient norm to this.
            min_seq_length (int): Skip instances whose history is shorter.
            early_stop (int): Stop if ``eval_metric`` hasn't improved for this
                many epochs. ``0`` disables early stopping.
            eval_metric (str): Metric to track for early stopping.
            metrics (Sequence[str]): Pointwise metrics reported each epoch.
            pairwise_metrics (Sequence[str]): Group-wise metrics reported each
                epoch.
            show_step (int): Log every N training mini-batches.
            save_model (bool): If True, checkpoint to ``model_dir``.
            save_epoch (int): Save every N epochs (only if ``save_model``).
            model_dir (str): Output directory for checkpoints.
        """
        if train_num_ngs < 1:
            raise ValueError("train_num_ngs must be >= 1.")
        if valid_num_ngs < 1:
            raise ValueError("valid_num_ngs must be >= 1.")

        self.train_num_ngs = train_num_ngs
        self.loss_name = loss
        self.embed_l2 = embed_l2
        self.embed_l1 = embed_l1
        self.layer_l2 = layer_l2
        self.layer_l1 = layer_l1
        self.optimizer = self._build_optimizer(optimizer, learning_rate)

        self.iterator = SequentialIterator(
            user_vocab=user_vocab,
            item_vocab=item_vocab,
            cate_vocab=cate_vocab,
            max_seq_length=self.max_seq_length,
            batch_size=batch_size,
        )

        best_metric = -float("inf")
        best_epoch = 0
        eval_info = []

        for epoch in range(1, epochs + 1):
            self.train()
            train_start = time.time()
            step = 0
            epoch_loss = 0.0

            for batch in self.iterator.load_data_from_file(
                train_file,
                min_seq_length=min_seq_length,
                batch_num_ngs=train_num_ngs,
            ):
                if batch is None:
                    continue
                t = self._to_device(batch)

                self.optimizer.zero_grad(set_to_none=True)
                logit = self.forward(
                    t["users"], t["items"], t["cates"], t["item_history"],
                    t["item_cate_history"], t["mask"],
                )
                data_loss = self._data_loss(logit, t["labels"])
                reg_loss = self._regular_loss()
                step_loss = data_loss + reg_loss

                step_loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()

                epoch_loss += step_loss.item()
                step += 1
                if step % show_step == 0:
                    logger.info(
                        "step %d , total_loss: %.4f, data_loss: %.4f",
                        step, step_loss.item(), data_loss.item(),
                    )

            train_time = time.time() - train_start
            valid_res = self.run_eval(
                valid_file, valid_num_ngs,
                metrics=metrics,
                pairwise_metrics=pairwise_metrics,
            )
            logger.info(
                "eval valid at epoch %d: %s",
                epoch,
                ",".join(f"{k}:{v}" for k, v in valid_res.items()),
            )
            eval_info.append((epoch, valid_res))

            progress = False
            if eval_metric in valid_res and valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                best_epoch = epoch
                progress = True
            elif early_stop > 0 and epoch - best_epoch >= early_stop:
                logger.info("early stop at epoch %d!", epoch)
                break

            if save_model and model_dir:
                os.makedirs(model_dir, exist_ok=True)
                if progress:
                    torch.save(
                        self.state_dict(),
                        os.path.join(model_dir, f"epoch_{epoch}_{MODEL_CHECKPOINT}"),
                    )
                    torch.save(
                        self.state_dict(),
                        os.path.join(model_dir, f"best_{MODEL_CHECKPOINT}"),
                    )

            logger.info("epoch %d train time: %.1fs", epoch, train_time)

        self.best_epoch = best_epoch
        return self

    # --- evaluation / inference ----------------------------------------------

    def run_eval(
        self,
        filename: str,
        num_ngs: int,
        metrics: Sequence[str] = DEFAULT_METRICS,
        pairwise_metrics: Sequence[str] = DEFAULT_PAIRWISE_METRICS,
    ) -> dict[str, float]:
        """Run pointwise + group-wise evaluation on ``filename``.

        Each row in ``filename`` is one positive or negative example; rows are
        grouped in chunks of ``num_ngs + 1`` (1 positive + ``num_ngs``
        negatives) so group-wise metrics (``group_auc``, ``mean_mrr``, ...)
        align with the TF model output.
        """
        if self.iterator is None:
            raise RuntimeError("run_eval() requires fit() to have been called.")
        group = num_ngs + 1

        self.eval()
        preds: list[float] = []
        labels: list[float] = []
        with torch.no_grad():
            for batch in self.iterator.load_data_from_file(
                filename, batch_num_ngs=0
            ):
                if batch is None:
                    continue
                t = self._to_device(batch)
                logit = self.forward(
                    t["users"], t["items"], t["cates"], t["item_history"],
                    t["item_cate_history"], t["mask"],
                )
                pred = torch.sigmoid(logit).reshape(-1).cpu().numpy()
                preds.extend(pred.tolist())
                labels.extend(batch["labels"].reshape(-1).tolist())

        res = cal_metric(labels, preds, list(metrics))
        group_labels = np.asarray(labels).reshape(-1, group).tolist()
        group_preds = np.asarray(preds).reshape(-1, group).tolist()
        res_pair = cal_metric(group_labels, group_preds, list(pairwise_metrics))
        res.update(res_pair)
        return res

    def predict(self, infile: str, outfile: str) -> "GRUModel":
        """Score every row of ``infile`` and write one score per line to ``outfile``."""
        if self.iterator is None:
            raise RuntimeError("predict() requires fit() to have been called.")
        self.eval()
        with open(outfile, "w") as wt, torch.no_grad():
            for batch in self.iterator.load_data_from_file(
                infile, batch_num_ngs=0
            ):
                if batch is None:
                    continue
                t = self._to_device(batch)
                logit = self.forward(
                    t["users"], t["items"], t["cates"], t["item_history"],
                    t["item_cate_history"], t["mask"],
                )
                pred = torch.sigmoid(logit).reshape(-1).cpu().numpy()
                wt.write("\n".join(str(x) for x in pred))
                wt.write("\n")
        return self

    def load(
        self,
        model_path: str,
        filename: str = f"best_{MODEL_CHECKPOINT}",
    ) -> None:
        """Load weights from a ``.pt`` file, or from ``model_path/<filename>`` if a directory.

        Args:
            model_path: Path to a ``.pt`` file, or a directory containing one.
            filename: Checkpoint name to load when ``model_path`` is a directory.
                Defaults to the best checkpoint written by ``fit(save_model=True)``;
                pass e.g. ``f"epoch_3_{MODEL_CHECKPOINT}"`` to load a specific epoch.
        """
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, filename)
        state_dict = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        self.load_state_dict(state_dict)
