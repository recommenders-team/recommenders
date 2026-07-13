# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Shared PyTorch building blocks and base class for deeprec sequential models.

Port of the reusable pieces of ``base_model.py`` / ``sequential_base_model.py`` that
SLi-Rec needs: the shared user/item/category embeddings, the soft-alignment
``Attention`` (long-term ASVD, unmasked), the ``FcnNet`` MLP head
(Linear -> BN -> Dropout -> Activation, matching the TF layer order), the softmax
pairwise ranking loss, the unique-embedding regularization, and the
``fit`` / ``run_eval`` / ``predict`` lifecycle. It mirrors the standalone
``nn.Module`` shape of ``graphrec/lightgcn.py`` and reuses ``deeprec_utils.cal_metric``
verbatim so reported numbers are directly comparable to the TF model.

Concrete numeric conventions preserved from TF:

* Model weights (embeddings, attention, FCN) use ``tnormal`` init
  (``trunc_normal_(std=init_value)``); biases are zero. The time-aware LSTM cell uses
  ``glorot_uniform`` (see :mod:`.rnn_cell_pytorch`).
* BatchNorm: TF ``momentum=0.95`` -> PyTorch ``momentum=0.05``; ``eps=1e-4``.
* Dropout is applied BEFORE the activation, only on hidden layers, only when
  ``user_dropout`` is set.
* Softmax loss ``= -group * mean(log(pos_softmax))`` over the full ``(N, group)``
  tensor (negatives replaced by 1 so their ``log`` is 0).
* Regularization uses ``tf.nn.l2_loss``'s ``0.5`` factor and only the UNIQUE involved
  item/category embeddings.
* Index 0 is both OOV and PAD with a trainable non-zero row -> no ``padding_idx``.
"""

from __future__ import annotations

import abc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.deeprec.deeprec_utils import cal_metric, load_dict
from recommenders.models.deeprec.io.sequential_dataset_pytorch import SequentialDataset

MODEL_CHECKPOINT = "best_model"

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "relu": F.relu,
    "tanh": torch.tanh,
    "elu": F.elu,
    "identity": lambda x: x,
}

_LONG_KEYS = ("users", "items", "cates", "item_history", "item_cate_history")
_FLOAT_KEYS = (
    "labels",
    "mask",
    "time",
    "time_diff",
    "time_from_first_action",
    "time_to_now",
)


def init_weight_(tensor: torch.Tensor, init_method: str, init_value: float) -> None:
    """Initialize a weight tensor to match the TF ``self.initializer``.

    Only ``tnormal`` (the SLi-Rec config) is reproduced exactly; other methods fall
    back to a sensible torch equivalent.
    """
    if init_method == "tnormal":
        nn.init.trunc_normal_(tensor, std=init_value)
    elif init_method == "normal":
        nn.init.normal_(tensor, std=init_value)
    elif init_method == "uniform":
        nn.init.uniform_(tensor, -init_value, init_value)
    elif init_method in ("xavier_normal", "xavier_uniform"):
        (
            nn.init.xavier_normal_
            if "normal" in init_method
            else nn.init.xavier_uniform_
        )(tensor)
    elif init_method in ("he_normal", "he_uniform"):
        (
            nn.init.kaiming_normal_
            if "normal" in init_method
            else nn.init.kaiming_uniform_
        )(tensor)
    else:
        nn.init.trunc_normal_(tensor, std=init_value)


class FcnNet(nn.Module):
    """MLP head matching TF ``_fcn_net``.

    Per hidden layer: ``Linear -> [BatchNorm] -> [Dropout] -> Activation``; a final
    ``Linear(-, 1)`` with no BN/dropout/activation. Works on 2-D ``[B, F]`` and 3-D
    ``[B, T, F]`` inputs (Linear/BN act on the last dimension).
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        activation: list[str],
        dropout: list[float],
        user_dropout: bool,
        enable_BN: bool,
        init_method: str,
        init_value: float,
    ) -> None:
        super().__init__()
        self.user_dropout = user_dropout
        self.enable_BN = enable_BN
        self.acts = [_ACTIVATIONS[a] for a in activation]

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        last = input_dim
        for idx, size in enumerate(layer_sizes):
            lin = nn.Linear(last, size)
            init_weight_(lin.weight, init_method, init_value)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)
            self.bns.append(
                nn.BatchNorm1d(size, momentum=0.05, eps=1e-4)
                if enable_BN
                else nn.Identity()
            )
            self.dropouts.append(nn.Dropout(p=dropout[idx]))
            last = size

        self.out = nn.Linear(last, 1)
        init_weight_(self.out.weight, init_method, init_value)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _apply_bn(bn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(bn, nn.Identity) or x.dim() == 2:
            return bn(x)
        b, t, f = x.shape
        return bn(x.reshape(-1, f)).reshape(b, t, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, lin in enumerate(self.linears):
            x = lin(x)
            x = self._apply_bn(self.bns[idx], x)
            if self.user_dropout:
                x = self.dropouts[idx](x)
            x = self.acts[idx](x)
        return self.out(x)


class Attention(nn.Module):
    """Long-term ASVD soft-alignment attention (unmasked), matching ``_attention``."""

    def __init__(
        self, input_dim: int, attention_size: int, init_method: str, init_value: float
    ) -> None:
        super().__init__()
        self.attention_mat = nn.Parameter(torch.empty(input_dim, input_dim))
        self.query = nn.Parameter(torch.empty(attention_size))
        init_weight_(self.attention_mat, init_method, init_value)
        init_weight_(self.query, init_method, init_value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs ``[B, T, E]`` -> weighted (not summed) inputs ``[B, T, E]``."""
        att_inputs = torch.einsum("bte,ef->btf", inputs, self.attention_mat)
        att_logits = torch.einsum("btf,f->bt", att_inputs, self.query)
        att_weights = torch.softmax(att_logits, dim=-1)
        return inputs * att_weights.unsqueeze(-1)


class SequentialBaseModel(nn.Module, abc.ABC):
    """Base class for PyTorch sequential recommenders (SLi-Rec and future ports).

    Every hyper-parameter is an explicit constructor / method argument — there is no
    ``hparams`` object — mirroring ``graphrec/lightgcn.py``: architecture on the
    constructor, training knobs on :meth:`fit`, evaluation knobs on :meth:`run_eval`.

    Subclasses set any model-specific architecture attributes, then call
    :meth:`_finalize` (after ``super().__init__``) to build the encoder and the MLP
    head with the correct RNG order.
    """

    def __init__(
        self,
        user_vocab: str,
        item_vocab: str,
        cate_vocab: str,
        max_seq_length: int = 50,
        item_embedding_dim: int = 32,
        cate_embedding_dim: int = 8,
        user_embedding_dim: int = 16,
        layer_sizes: list[int] | None = None,
        activation: list[str] | None = None,
        dropout: list[float] | None = None,
        user_dropout: bool = True,
        enable_BN: bool = True,
        init_method: str = "tnormal",
        init_value: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.cate_vocab = cate_vocab
        self.max_seq_length = max_seq_length
        self.item_embedding_dim = item_embedding_dim
        self.cate_embedding_dim = cate_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.layer_sizes = layer_sizes if layer_sizes is not None else [100, 64]
        self.activation = activation if activation is not None else ["relu", "relu"]
        self.dropout = dropout if dropout is not None else [0.3, 0.3]
        self.user_dropout = user_dropout
        self.enable_BN = enable_BN
        self.init_method = init_method
        self.init_value = init_value
        self.min_seq_length = 1

        self.iterator = SequentialDataset(
            user_vocab, item_vocab, cate_vocab, max_seq_length
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_embedding()

        # Training/eval defaults; fit()/run_eval() override these.
        self.optimizer = None
        self.best_epoch = 0
        self.batch_size = 400
        self.train_num_ngs = 4
        self.embed_l2 = 0.0
        self.embed_l1 = 0.0
        self.layer_l2 = 0.0
        self.layer_l1 = 0.0
        self.metrics = ["auc", "logloss"]
        self.pairwise_metrics = ["mean_mrr", "ndcg@2;4;6", "group_auc"]

    def _finalize(self) -> None:
        """Build the encoder and MLP head, then move to device.

        Called by subclasses after ``super().__init__`` and after they set their
        model-specific architecture attributes.
        """
        model_output_dim = self._build_seq_graph()
        self.logit_fcn = FcnNet(
            model_output_dim,
            self.layer_sizes,
            self.activation,
            self.dropout,
            self.user_dropout,
            self.enable_BN,
            self.init_method,
            self.init_value,
        )
        self.to(self.device)

    def _build_embedding(self) -> None:
        self.user_vocab_length = len(load_dict(self.user_vocab))
        self.item_vocab_length = len(load_dict(self.item_vocab))
        self.cate_vocab_length = len(load_dict(self.cate_vocab))

        self.user_lookup = nn.Embedding(self.user_vocab_length, self.user_embedding_dim)
        self.item_lookup = nn.Embedding(self.item_vocab_length, self.item_embedding_dim)
        self.cate_lookup = nn.Embedding(self.cate_vocab_length, self.cate_embedding_dim)
        for emb in (self.user_lookup, self.item_lookup, self.cate_lookup):
            init_weight_(emb.weight, self.init_method, self.init_value)

    @abc.abstractmethod
    def _build_seq_graph(self) -> int:
        """Create the sequence-encoder submodules; return the model_output feature dim."""

    @abc.abstractmethod
    def _seq_forward(self, batch: dict) -> torch.Tensor:
        """Compute ``model_output`` ``[B, D]`` from a batch of tensors + embeddings."""

    def _lookup(self, batch: dict) -> None:
        """Populate embedding attributes used by ``_seq_forward`` (TF-style)."""
        self.user_embedding = self.user_lookup(batch["users"])
        self.item_embedding = self.item_lookup(batch["items"])
        self.cate_embedding = self.cate_lookup(batch["cates"])
        self.item_history_embedding = self.item_lookup(batch["item_history"])
        self.cate_history_embedding = self.cate_lookup(batch["item_cate_history"])
        self.target_item_embedding = torch.cat(
            [self.item_embedding, self.cate_embedding], dim=-1
        )

    def forward(self, batch: dict) -> torch.Tensor:
        self._lookup(batch)
        model_output = self._seq_forward(batch)
        return self.logit_fcn(model_output)

    def _to_tensors(self, np_batch: dict) -> dict:
        out = {}
        for k in _LONG_KEYS:
            out[k] = torch.as_tensor(np_batch[k], dtype=torch.long, device=self.device)
        for k in _FLOAT_KEYS:
            out[k] = torch.as_tensor(
                np_batch[k], dtype=torch.float32, device=self.device
            )
        return out

    def _softmax_loss(self, logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        group = self.train_num_ngs + 1
        logits = logit.view(-1, group)
        labels = labels.view(-1, group)
        softmax_pred = torch.softmax(logits, dim=-1)
        pos_softmax = torch.where(
            labels == 1.0, softmax_pred, torch.ones_like(softmax_pred)
        )
        return -group * torch.mean(torch.log(pos_softmax))

    def _regular_loss(self, batch: dict) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        if self.embed_l2 > 0 or self.embed_l1 > 0:
            involved_items = torch.unique(
                torch.cat([batch["item_history"].reshape(-1), batch["items"]])
            )
            involved_cates = torch.unique(
                torch.cat([batch["item_cate_history"].reshape(-1), batch["cates"]])
            )
            item_e = self.item_lookup(involved_items)
            cate_e = self.cate_lookup(involved_cates)
            if self.embed_l2 > 0:
                reg = reg + self.embed_l2 * 0.5 * (
                    item_e.pow(2).sum() + cate_e.pow(2).sum()
                )
            if self.embed_l1 > 0:
                reg = reg + self.embed_l1 * (item_e.abs().sum() + cate_e.abs().sum())
        if self.layer_l2 > 0 or self.layer_l1 > 0:
            emb_params = set(
                id(p)
                for m in (self.user_lookup, self.item_lookup, self.cate_lookup)
                for p in m.parameters()
            )
            for p in self.parameters():
                if id(p) in emb_params:
                    continue
                if self.layer_l2 > 0:
                    reg = reg + self.layer_l2 * 0.5 * p.pow(2).sum()
                if self.layer_l1 > 0:
                    reg = reg + self.layer_l1 * p.abs().sum()
        return reg

    def fit(
        self,
        train_file: str,
        valid_file: str,
        epochs: int = 50,
        batch_size: int = 400,
        learning_rate: float = 0.001,
        train_num_ngs: int = 4,
        valid_num_ngs: int = 4,
        embed_l2: float = 0.0,
        layer_l2: float = 0.0,
        embed_l1: float = 0.0,
        layer_l1: float = 0.0,
        is_clip_norm: bool = False,
        max_grad_norm: float = 2.0,
        eval_metric: str = "group_auc",
        metrics: list[str] | None = None,
        pairwise_metrics: list[str] | None = None,
        show_step: int = 100,
        save_model: bool = False,
        model_dir: str | None = None,
        early_stop: int = 10,
    ):
        """Train, evaluating on ``valid_file`` each epoch with early stopping.

        Args:
            train_file (str): Training data (positives only when sampling negatives).
            valid_file (str): Validation data, grouped as ``1 + valid_num_ngs``.
            epochs (int): Number of training epochs.
            batch_size (int): Positive instances per mini-batch.
            learning_rate (float): Adam learning rate.
            train_num_ngs (int): In-batch negatives sampled per positive (loss group
                size is ``train_num_ngs + 1``).
            valid_num_ngs (int): Negatives per positive in ``valid_file``.
            embed_l2, layer_l2, embed_l1, layer_l1 (float): Regularization coefficients.
            is_clip_norm (bool): Enable per-parameter gradient-norm clipping.
            max_grad_norm (float): Clip value when ``is_clip_norm`` is set.
            eval_metric (str): Validation metric driving early stopping / best model.
            metrics (list[str]): Pointwise metrics. Defaults to ``["auc", "logloss"]``.
            pairwise_metrics (list[str]): Group metrics. Defaults to
                ``["mean_mrr", "ndcg@2;4;6", "group_auc"]``.
            show_step (int): Print training loss every ``show_step`` steps.
            save_model (bool): Save the best-by-validation model under ``model_dir``.
            model_dir (str): Directory for the ``best_model`` checkpoint.
            early_stop (int): Stop if the metric does not improve for this many epochs.
        """
        self.batch_size = batch_size
        self.train_num_ngs = train_num_ngs
        self.embed_l2, self.layer_l2 = embed_l2, layer_l2
        self.embed_l1, self.layer_l1 = embed_l1, layer_l1
        if metrics is not None:
            self.metrics = metrics
        if pairwise_metrics is not None:
            self.pairwise_metrics = pairwise_metrics

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_metric, self.best_epoch = 0, 0

        for epoch in range(1, epochs + 1):
            self.train()
            step = 0
            for np_batch in self.iterator.load_data_from_file(
                train_file,
                batch_size=batch_size,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=train_num_ngs,
            ):
                if not np_batch:
                    continue
                batch = self._to_tensors(np_batch)
                self.optimizer.zero_grad(set_to_none=True)
                logit = self.forward(batch)
                data_loss = self._softmax_loss(logit, batch["labels"])
                loss = data_loss + self._regular_loss(batch)
                loss.backward()
                if is_clip_norm:
                    for p in self.parameters():
                        if p.grad is not None:
                            nn.utils.clip_grad_norm_(p, max_grad_norm)
                self.optimizer.step()
                step += 1
                if step % show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, loss.item(), data_loss.item()
                        )
                    )

            valid_res = self.run_eval(valid_file, valid_num_ngs, batch_size=batch_size)
            print("eval valid at epoch {0}: {1}".format(epoch, valid_res))

            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                if save_model and model_dir:
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(
                        self.state_dict(), os.path.join(model_dir, MODEL_CHECKPOINT)
                    )
            elif early_stop > 0 and epoch - self.best_epoch >= early_stop:
                print("early stop at epoch {0}!".format(epoch))
                break

        print("best epoch: {0}".format(self.best_epoch))
        return self

    @torch.no_grad()
    def run_eval(
        self,
        filename: str,
        num_ngs: int,
        batch_size: int | None = None,
        metrics: list[str] | None = None,
        pairwise_metrics: list[str] | None = None,
    ) -> dict:
        """Evaluate ``filename``; returns the metric dict (pointwise + pairwise).

        Args:
            filename (str): Evaluation file, grouped as ``1 + num_ngs`` consecutive rows.
            num_ngs (int): Negatives per positive in ``filename``.
            batch_size (int): Batch size; defaults to the value used in ``fit``.
            metrics, pairwise_metrics (list[str]): Override the reported metrics.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        metrics = metrics if metrics is not None else self.metrics
        pairwise_metrics = (
            pairwise_metrics if pairwise_metrics is not None else self.pairwise_metrics
        )
        self.eval()
        preds, labels, group_preds, group_labels = [], [], [], []
        group = num_ngs + 1
        for np_batch in self.iterator.load_data_from_file(
            filename,
            batch_size=batch_size,
            min_seq_length=self.min_seq_length,
            batch_num_ngs=0,
        ):
            if not np_batch:
                continue
            batch = self._to_tensors(np_batch)
            logit = self.forward(batch)
            pred = torch.sigmoid(logit).cpu().numpy()
            step_labels = np_batch["labels"]
            preds.extend(np.reshape(pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            group_preds.extend(np.reshape(pred, (-1, group)))
            group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, metrics)
        res_pairwise = cal_metric(group_labels, group_preds, pairwise_metrics)
        res.update(res_pairwise)
        return res

    @torch.no_grad()
    def predict(
        self, infile_name: str, outfile_name: str, batch_size: int | None = None
    ):
        """Write per-instance prediction scores (one per line) to ``outfile_name``."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.eval()
        with open(outfile_name, "w") as wt:
            for np_batch in self.iterator.load_data_from_file(
                infile_name,
                batch_size=batch_size,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=0,
            ):
                if not np_batch:
                    continue
                batch = self._to_tensors(np_batch)
                pred = torch.sigmoid(self.forward(batch)).cpu().numpy()
                pred = np.reshape(pred, -1)
                wt.write("\n".join(map(str, pred)))
                wt.write("\n")
        return self

    def load_model(self, model_path: str):
        """Restore parameters from a ``state_dict`` checkpoint.

        Args:
            model_path (str): Path to the checkpoint file, or a directory containing
                the ``best_model`` checkpoint.
        """
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, MODEL_CHECKPOINT)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state)
        return self
