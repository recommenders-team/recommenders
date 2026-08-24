# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""PyTorch port of xDeepFM (eXtreme Deep Factorization Machine).

Reproduces the TF-1.x ``XDeepFMModel`` and the parts of ``base_model.py`` it used,
as a standalone ``nn.Module``. The four components -- linear regression, a 2-order
FM, the Compressed Interaction Network (CIN) and a plain DNN -- are enabled
independently and their logits are summed, exactly as in the TF ``_build_graph``.

Concrete conventions preserved from TF:

* All four components read one shared ``[feature_count, dim]`` embedding table.
  The linear and FM parts sum over a whole instance; the CIN and DNN parts sum per
  field, which is what ``embedding_lookup_sparse(combiner="sum")`` did.
* ``tf.nn.l2_loss`` has a ``0.5`` factor, kept here for ``embed_l2`` / ``layer_l2``,
  while ``cross_l2`` stays the unsquared Frobenius norm that ``tf.norm(..., ord=2)``
  computed.
* ``log_loss`` uses TF's ``1e-7`` epsilon; ``square_loss`` is an RMSE.

Two dead code paths from the TF model are not ported: ``_build_fast_CIN`` (a
TF-op-level parameter-decomposition speed hack, disabled by ``fast_CIN_d: 0``
everywhere) and the ``res`` / ``direct`` / ``bias`` / ``is_masked`` switches of
``_build_CIN``, which ``_build_graph`` hardcoded to a single combination.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.deeprec.io.ffm_dataset import FFMDataset
from recommenders.models.deeprec.models.pytorch.layers import (
    ACTIVATIONS,
    FcnNet,
    apply_bn,
    init_weight_,
)

__all__ = ["CIN", "XDeepFMModel"]


class CIN(nn.Module):
    """Compressed Interaction Network, the explicit vector-wise component.

    Port of ``_build_CIN`` in the one configuration ``_build_graph`` ever built it:
    residual output, hidden units split in half (one half fed to the next layer, the
    other to the output), no bias term, and the first layer masked so that a field
    never interacts with itself.
    """

    def __init__(
        self,
        field_count: int,
        dim: int,
        layer_sizes: list[int],
        activation: str,
        enable_BN: bool,
        init_method: str,
        init_value: float,
    ) -> None:
        super().__init__()
        for layer_size in layer_sizes[:-1]:
            if layer_size % 2 != 0:
                raise ValueError(
                    "Every cross layer but the last is split in half, so its size "
                    "must be even; got {0}.".format(layer_sizes)
                )

        self.dim = dim
        self.act = ACTIVATIONS[activation]

        self.filters = nn.ParameterList()
        self.bns = nn.ModuleList()
        field_nums = [field_count]
        final_len = 0
        for idx, layer_size in enumerate(layer_sizes):
            filters = nn.Parameter(
                torch.empty(field_nums[0] * field_nums[-1], layer_size)
            )
            init_weight_(filters, init_method, init_value)
            self.filters.append(filters)
            self.bns.append(
                nn.BatchNorm1d(layer_size, momentum=0.05, eps=1e-4)
                if enable_BN
                else nn.Identity()
            )
            if idx != len(layer_sizes) - 1:
                field_nums.append(layer_size // 2)
                final_len += layer_size // 2
            else:
                final_len += layer_size

        self.w_out = nn.Parameter(torch.empty(final_len, 1))
        init_weight_(self.w_out, init_method, init_value)
        self.b_out = nn.Parameter(torch.zeros(1))

        # Strictly upper triangular: drops self-interactions in the first layer, and
        # the doubling compensates for the symmetric half dropped along with them.
        mask = torch.triu(torch.ones(field_count, field_count), diagonal=1)
        self.register_buffer("mask", mask.reshape(1, 1, -1) * 2)

    def forward(self, field_embed: torch.Tensor) -> torch.Tensor:
        """field_embed ``[B, F, D]`` -> logit ``[B, 1]``."""
        x_0 = field_embed
        x_k = field_embed
        direct_connects = []
        for idx, filters in enumerate(self.filters):
            # Outer product of every (x_0, x_k) field pair, per embedding dimension.
            interactions = torch.einsum("bfd,bgd->bdfg", x_0, x_k)
            interactions = interactions.reshape(x_0.shape[0], self.dim, -1)
            if idx == 0:
                interactions = interactions * self.mask

            out = interactions @ filters
            out = apply_bn(self.bns[idx], out)
            out = self.act(out).transpose(1, 2)

            if idx != len(self.filters) - 1:
                x_k, direct_connect = torch.split(out, out.shape[1] // 2, dim=1)
            else:
                direct_connect = out
            direct_connects.append(direct_connect)

        result = torch.cat(direct_connects, dim=1).sum(dim=-1)
        return result.sum(dim=1, keepdim=True) + result @ self.w_out + self.b_out


class XDeepFMModel(nn.Module):
    """xDeepFM model (PyTorch).

    :Citation:

        J. Lian, X. Zhou, F. Zhang, Z. Chen, X. Xie, G. Sun, "xDeepFM: Combining Explicit
        and Implicit Feature Interactions for Recommender Systems", in Proceedings of the
        24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining,
        KDD 2018, London, 2018.
    """

    def __init__(
        self,
        feature_count: int,
        field_count: int,
        dim: int = 10,
        method: str = "classification",
        use_linear_part: bool = False,
        use_fm_part: bool = False,
        use_cin_part: bool = False,
        use_dnn_part: bool = False,
        cross_layer_sizes: list[int] | None = None,
        cross_activation: str = "identity",
        layer_sizes: list[int] | None = None,
        activation: list[str] | None = None,
        dropout: list[float] | None = None,
        user_dropout: bool = False,
        enable_BN: bool = False,
        init_method: str = "tnormal",
        init_value: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Build the PyTorch xDeepFM model.

        Architecture arguments live on the constructor; training-time knobs (epochs,
        learning rate, batch size, loss, regularization, ...) belong on :meth:`fit`.
        Enabling only ``use_linear_part`` and ``use_fm_part`` gives a classical FM.

        Args:
            feature_count (int): Size of the shared feature embedding table.
            field_count (int): Number of fields per instance.
            dim (int): Feature embedding dimension.
            method (str): ``classification`` (sigmoid output) or ``regression``.
            use_linear_part (bool): Add the linear-regression logit.
            use_fm_part (bool): Add the 2-order factorization-machine logit.
            use_cin_part (bool): Add the Compressed Interaction Network logit.
            use_dnn_part (bool): Add the DNN logit.
            cross_layer_sizes (list[int]): CIN layer sizes. Every size but the last
                must be even. Defaults to ``[100, 100]``.
            cross_activation (str): Activation inside the CIN.
            layer_sizes (list[int]): DNN layer sizes. Defaults to ``[100, 100]``.
            activation (list[str]): Per-layer DNN activations. Defaults to
                ``["relu", "relu"]``.
            dropout (list[float]): Per-layer DNN dropout rates, one per layer.
            user_dropout (bool): Whether to apply dropout in the DNN.
            enable_BN (bool): Whether to use batch normalization in the CIN and DNN.
            init_method (str): Weight init scheme (``tnormal`` by default).
            init_value (float): Std/scale for weight initialization.
            seed (int): Random seed.
        """
        super().__init__()
        if method not in ("classification", "regression"):
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    method
                )
            )
        if not (use_linear_part or use_fm_part or use_cin_part or use_dnn_part):
            raise ValueError(
                "At least one of use_linear_part, use_fm_part, use_cin_part or "
                "use_dnn_part must be enabled."
            )

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.feature_count = feature_count
        self.field_count = field_count
        self.dim = dim
        self.method = method
        self.use_linear_part = use_linear_part
        self.use_fm_part = use_fm_part
        self.use_cin_part = use_cin_part
        self.use_dnn_part = use_dnn_part

        self.iterator = FFMDataset(field_count)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Parameter(torch.empty(feature_count, dim))
        init_weight_(self.embedding, init_method, init_value)
        self.embed_params = [self.embedding]
        self.layer_params = []
        self.cross_params = []

        if use_linear_part:
            self.linear_w = nn.Parameter(torch.empty(feature_count, 1))
            init_weight_(self.linear_w, init_method, init_value)
            self.linear_b = nn.Parameter(torch.zeros(1))
            self.layer_params += [self.linear_w, self.linear_b]

        if use_cin_part:
            self.cin = CIN(
                field_count,
                dim,
                cross_layer_sizes if cross_layer_sizes is not None else [100, 100],
                cross_activation,
                enable_BN,
                init_method,
                init_value,
            )
            self.cross_params += list(self.cin.filters)
            self.layer_params += [self.cin.w_out, self.cin.b_out]

        if use_dnn_part:
            self.dnn = FcnNet(
                field_count * dim,
                layer_sizes if layer_sizes is not None else [100, 100],
                activation if activation is not None else ["relu", "relu"],
                dropout if dropout is not None else [0.0, 0.0],
                user_dropout,
                enable_BN,
                init_method,
                init_value,
            )
            # TF only put the Linear weights in layer_params: the batch-norm gammas
            # and betas came from tf.layers.batch_normalization and stayed
            # unregularized.
            self.layer_params += [
                param
                for module in (*self.dnn.linears, self.dnn.out)
                for param in module.parameters()
            ]

        # Training defaults; fit() and run_eval() override these.
        self.batch_size = 128
        self.embed_l2 = 0.0
        self.embed_l1 = 0.0
        self.layer_l2 = 0.0
        self.layer_l1 = 0.0
        self.cross_l2 = 0.0
        self.cross_l1 = 0.0
        self.metrics = ["auc", "logloss"]

        self.to(self.device)

    def forward(self, batch: dict) -> torch.Tensor:
        """Sum the logits of every enabled component; returns ``[B, 1]``."""
        feat_ids = batch["feat_ids"]
        feat_values = batch["feat_values"]
        dnn_offsets = batch["dnn_offsets"]
        # Every field_count-th field bag starts a new instance.
        instance_offsets = dnn_offsets[:: self.field_count].contiguous()

        logit = torch.zeros(
            instance_offsets.shape[0], 1, device=feat_values.device, dtype=torch.float32
        )

        if self.use_linear_part:
            logit = (
                logit
                + F.embedding_bag(
                    feat_ids,
                    self.linear_w,
                    instance_offsets,
                    mode="sum",
                    per_sample_weights=feat_values,
                )
                + self.linear_b
            )

        if self.use_fm_part:
            summed = F.embedding_bag(
                feat_ids,
                self.embedding,
                instance_offsets,
                mode="sum",
                per_sample_weights=feat_values,
            )
            squared = F.embedding_bag(
                feat_ids,
                self.embedding.pow(2),
                instance_offsets,
                mode="sum",
                per_sample_weights=feat_values.pow(2),
            )
            logit = logit + 0.5 * (summed.pow(2) - squared).sum(dim=1, keepdim=True)

        if self.use_cin_part or self.use_dnn_part:
            # Sum pooling per (instance, field): [B * field_count, dim].
            field_embed = F.embedding_bag(
                feat_ids,
                self.embedding,
                dnn_offsets,
                mode="sum",
                per_sample_weights=feat_values,
            )
            if self.use_cin_part:
                logit = logit + self.cin(
                    field_embed.view(-1, self.field_count, self.dim)
                )
            if self.use_dnn_part:
                logit = logit + self.dnn(
                    field_embed.view(-1, self.field_count * self.dim)
                )

        return logit

    def _get_pred(self, logit: torch.Tensor) -> torch.Tensor:
        """Turn the logit into a prediction score according to ``method``."""
        return torch.sigmoid(logit) if self.method == "classification" else logit

    def _to_tensors(self, np_batch: dict) -> dict:
        return {
            "labels": torch.as_tensor(
                np_batch["labels"], dtype=torch.float32, device=self.device
            ),
            "feat_ids": torch.as_tensor(
                np_batch["feat_ids"], dtype=torch.long, device=self.device
            ),
            "feat_values": torch.as_tensor(
                np_batch["feat_values"], dtype=torch.float32, device=self.device
            ),
            "dnn_offsets": torch.as_tensor(
                np_batch["dnn_offsets"], dtype=torch.long, device=self.device
            ),
        }

    def _data_loss(
        self,
        logit: torch.Tensor,
        pred: torch.Tensor,
        labels: torch.Tensor,
        loss: str,
    ) -> torch.Tensor:
        logit, pred, labels = logit.view(-1), pred.view(-1), labels.view(-1)
        if loss == "cross_entropy_loss":
            return F.binary_cross_entropy_with_logits(logit, labels)
        elif loss == "square_loss":
            return torch.sqrt(F.mse_loss(pred, labels))
        elif loss == "log_loss":
            epsilon = 1e-7
            return torch.mean(
                -labels * torch.log(pred + epsilon)
                - (1 - labels) * torch.log(1 - pred + epsilon)
            )
        raise ValueError("this loss not defined {0}".format(loss))

    def _regular_loss(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        if self.embed_l2 > 0 or self.embed_l1 > 0:
            for param in self.embed_params:
                reg = reg + self.embed_l2 * 0.5 * param.pow(2).sum()
                reg = reg + self.embed_l1 * param.abs().sum()
        if self.layer_l2 > 0 or self.layer_l1 > 0:
            for param in self.layer_params:
                reg = reg + self.layer_l2 * 0.5 * param.pow(2).sum()
                reg = reg + self.layer_l1 * param.abs().sum()
        if self.cross_l2 > 0 or self.cross_l1 > 0:
            for param in self.cross_params:
                reg = reg + self.cross_l1 * param.abs().sum()
                # tf.norm(..., ord=2) is the Frobenius norm, not the squared one.
                reg = reg + self.cross_l2 * param.pow(2).sum().sqrt()
        return reg

    def fit(
        self,
        train_file: str,
        valid_file: str,
        test_file: str | None = None,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        loss: str = "cross_entropy_loss",
        embed_l2: float = 0.0,
        embed_l1: float = 0.0,
        layer_l2: float = 0.0,
        layer_l1: float = 0.0,
        cross_l2: float = 0.0,
        cross_l1: float = 0.0,
        is_clip_norm: bool = False,
        max_grad_norm: float = 2.0,
        metrics: list[str] | None = None,
        show_step: int = 1,
        save_model: bool = False,
        model_dir: str | None = None,
        save_epoch: int = 5,
    ):
        """Train on ``train_file``, evaluating on ``valid_file`` after every epoch.

        Args:
            train_file (str): Training data set.
            valid_file (str): Validation set, evaluated every epoch.
            test_file (str): Optional test set, also evaluated every epoch.
            epochs (int): Number of training epochs.
            batch_size (int): Instances per mini-batch.
            learning_rate (float): Adam learning rate.
            loss (str): ``cross_entropy_loss``, ``log_loss`` or ``square_loss``.
            embed_l2, embed_l1 (float): Regularization on the embedding table.
            layer_l2, layer_l1 (float): Regularization on the linear, DNN and
                CIN-output parameters.
            cross_l2, cross_l1 (float): Regularization on the CIN filters.
            is_clip_norm (bool): Enable per-parameter gradient-norm clipping.
            max_grad_norm (float): Clip value when ``is_clip_norm`` is set.
            metrics (list[str]): Metrics to report. Defaults to ``["auc", "logloss"]``.
            show_step (int): Print the training loss every ``show_step`` steps.
            save_model (bool): Save a checkpoint under ``model_dir``.
            model_dir (str): Directory for the ``epoch_<n>`` checkpoints.
            save_epoch (int): Save a checkpoint every ``save_epoch`` epochs.

        Returns:
            object: An instance of self.
        """
        self.batch_size = batch_size
        self.embed_l2, self.embed_l1 = embed_l2, embed_l1
        self.layer_l2, self.layer_l1 = layer_l2, layer_l1
        self.cross_l2, self.cross_l1 = cross_l2, cross_l1
        if metrics is not None:
            self.metrics = metrics

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            self.train()
            step = 0
            epoch_loss = 0.0
            for np_batch, _, _ in self.iterator.load_data_from_file(
                train_file, batch_size
            ):
                batch = self._to_tensors(np_batch)
                optimizer.zero_grad(set_to_none=True)
                logit = self.forward(batch)
                data_loss = self._data_loss(
                    logit, self._get_pred(logit), batch["labels"], loss
                )
                step_loss = data_loss + self._regular_loss()
                step_loss.backward()
                if is_clip_norm:
                    for param in self.parameters():
                        if param.grad is not None:
                            nn.utils.clip_grad_norm_(param, max_grad_norm)
                optimizer.step()

                epoch_loss += step_loss.item()
                step += 1
                if step % show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss.item(), data_loss.item()
                        )
                    )

            if save_model and model_dir and epoch % save_epoch == 0:
                os.makedirs(model_dir, exist_ok=True)
                torch.save(
                    self.state_dict(), os.path.join(model_dir, "epoch_" + str(epoch))
                )

            train_info = "logloss loss:{0}".format(epoch_loss / step)
            eval_info = self._format_metrics(self.run_eval(valid_file, batch_size))
            message = "at epoch {0:d}\ntrain info: {1}\neval info: {2}".format(
                epoch, train_info, eval_info
            )
            if test_file is not None:
                message += "\ntest info: " + self._format_metrics(
                    self.run_eval(test_file, batch_size)
                )
            print(message)

        return self

    @staticmethod
    def _format_metrics(res: dict) -> str:
        return ", ".join(
            "{0}:{1}".format(name, value) for name, value in sorted(res.items())
        )

    @torch.no_grad()
    def run_eval(
        self,
        filename: str,
        batch_size: int | None = None,
        metrics: list[str] | None = None,
    ) -> dict:
        """Evaluate ``filename`` and return the metric dictionary.

        Args:
            filename (str): A file name that will be evaluated.
            batch_size (int): Batch size; defaults to the value used in ``fit``.
            metrics (list[str]): Override the reported metrics.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        metrics = metrics if metrics is not None else self.metrics
        self.eval()
        preds, labels = [], []
        for np_batch, _, _ in self.iterator.load_data_from_file(filename, batch_size):
            batch = self._to_tensors(np_batch)
            pred = self._get_pred(self.forward(batch)).cpu().numpy()
            preds.extend(np.reshape(pred, -1))
            labels.extend(np.reshape(np_batch["labels"], -1))
        return cal_metric(labels, preds, metrics)

    @torch.no_grad()
    def predict(
        self, infile_name: str, outfile_name: str, batch_size: int | None = None
    ):
        """Write the prediction score of every instance, one per line.

        Args:
            infile_name (str): Input file name, format is same as train/val/test file.
            outfile_name (str): Output file name, each line is the predict score.
            batch_size (int): Batch size; defaults to the value used in ``fit``.

        Returns:
            object: An instance of self.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.eval()
        with open(outfile_name, "w") as wt:
            for np_batch, _, _ in self.iterator.load_data_from_file(
                infile_name, batch_size
            ):
                batch = self._to_tensors(np_batch)
                pred = self._get_pred(self.forward(batch)).cpu().numpy()
                wt.write("\n".join(map(str, np.reshape(pred, -1))))
                wt.write("\n")
        return self

    def load_model(self, model_path: str):
        """Restore parameters from a ``state_dict`` checkpoint.

        Args:
            model_path (str): Path to the checkpoint file.

        Returns:
            object: An instance of self.
        """
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state)
        return self
