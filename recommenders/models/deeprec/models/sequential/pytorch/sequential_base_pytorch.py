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
    """Base class for PyTorch sequential recommenders (SLi-Rec and future ports)."""

    def __init__(self, hparams, iterator_creator, seed: int | None = None) -> None:
        super().__init__()
        self.hparams = hparams
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.iterator = iterator_creator(hparams)
        self.train_num_ngs = hparams.train_num_ngs
        self.min_seq_length = 1

        self.init_method = hparams.init_method
        self.init_value = hparams.init_value

        self._build_embedding()
        model_output_dim = self._build_seq_graph()
        self.logit_fcn = FcnNet(
            model_output_dim,
            hparams.layer_sizes,
            hparams.activation,
            hparams.dropout,
            hparams.user_dropout,
            hparams.enable_BN,
            self.init_method,
            self.init_value,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer: torch.optim.Optimizer | None = None
        self.best_epoch = 0

    def _build_embedding(self) -> None:
        h = self.hparams
        self.user_vocab_length = len(load_dict(h.user_vocab))
        self.item_vocab_length = len(load_dict(h.item_vocab))
        self.cate_vocab_length = len(load_dict(h.cate_vocab))
        self.item_embedding_dim = h.item_embedding_dim
        self.cate_embedding_dim = h.cate_embedding_dim
        self.user_embedding_dim = h.user_embedding_dim

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
        h = self.hparams
        reg = torch.zeros((), device=self.device)
        if h.embed_l2 > 0 or h.embed_l1 > 0:
            involved_items = torch.unique(
                torch.cat([batch["item_history"].reshape(-1), batch["items"]])
            )
            involved_cates = torch.unique(
                torch.cat([batch["item_cate_history"].reshape(-1), batch["cates"]])
            )
            item_e = self.item_lookup(involved_items)
            cate_e = self.cate_lookup(involved_cates)
            if h.embed_l2 > 0:
                reg = reg + h.embed_l2 * 0.5 * (
                    item_e.pow(2).sum() + cate_e.pow(2).sum()
                )
            if h.embed_l1 > 0:
                reg = reg + h.embed_l1 * (item_e.abs().sum() + cate_e.abs().sum())
        if h.layer_l2 > 0 or h.layer_l1 > 0:
            emb_params = set(
                id(p)
                for m in (self.user_lookup, self.item_lookup, self.cate_lookup)
                for p in m.parameters()
            )
            for p in self.parameters():
                if id(p) in emb_params:
                    continue
                if h.layer_l2 > 0:
                    reg = reg + h.layer_l2 * 0.5 * p.pow(2).sum()
                if h.layer_l1 > 0:
                    reg = reg + h.layer_l1 * p.abs().sum()
        return reg

    def fit(
        self,
        train_file: str,
        valid_file: str,
        valid_num_ngs: int,
        eval_metric="group_auc",
    ):
        """Train, evaluating on ``valid_file`` each epoch with early stopping."""
        h = self.hparams
        self.optimizer = torch.optim.Adam(self.parameters(), lr=h.learning_rate)
        best_metric, self.best_epoch = 0, 0

        for epoch in range(1, h.epochs + 1):
            self.train()
            step, epoch_loss = 0, 0.0
            for np_batch in self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            ):
                if not np_batch:
                    continue
                batch = self._to_tensors(np_batch)
                self.optimizer.zero_grad(set_to_none=True)
                logit = self.forward(batch)
                data_loss = self._softmax_loss(logit, batch["labels"])
                loss = data_loss + self._regular_loss(batch)
                loss.backward()
                if h.is_clip_norm:
                    for p in self.parameters():
                        if p.grad is not None:
                            nn.utils.clip_grad_norm_(p, h.max_grad_norm)
                self.optimizer.step()
                epoch_loss += loss.item()
                step += 1
                if step % h.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, loss.item(), data_loss.item()
                        )
                    )

            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print("eval valid at epoch {0}: {1}".format(epoch, valid_res))

            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                if h.save_model and h.MODEL_DIR:
                    os.makedirs(h.MODEL_DIR, exist_ok=True)
                    torch.save(
                        self.state_dict(), os.path.join(h.MODEL_DIR, MODEL_CHECKPOINT)
                    )
            elif h.EARLY_STOP > 0 and epoch - self.best_epoch >= h.EARLY_STOP:
                print("early stop at epoch {0}!".format(epoch))
                break

        print("best epoch: {0}".format(self.best_epoch))
        return self

    @torch.no_grad()
    def run_eval(self, filename: str, num_ngs: int) -> dict:
        """Evaluate ``filename``; returns the metric dict (pointwise + pairwise)."""
        self.eval()
        preds, labels, group_preds, group_labels = [], [], [], []
        group = num_ngs + 1
        for np_batch in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
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

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    @torch.no_grad()
    def predict(self, infile_name: str, outfile_name: str):
        """Write per-instance prediction scores (one per line) to ``outfile_name``."""
        self.eval()
        with open(outfile_name, "w") as wt:
            for np_batch in self.iterator.load_data_from_file(
                infile_name, min_seq_length=self.min_seq_length, batch_num_ngs=0
            ):
                if not np_batch:
                    continue
                batch = self._to_tensors(np_batch)
                pred = torch.sigmoid(self.forward(batch)).cpu().numpy()
                pred = np.reshape(pred, -1)
                wt.write("\n".join(map(str, pred)))
                wt.write("\n")
        return self

    def load_model(self, model_path: str | None = None):
        """Restore parameters from a ``state_dict`` checkpoint."""
        if model_path is None:
            model_path = os.path.join(self.hparams.MODEL_DIR, MODEL_CHECKPOINT)
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, MODEL_CHECKPOINT)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state)
        return self
