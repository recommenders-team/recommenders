# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""PyTorch port of SLi-Rec (Adaptive User Modeling with Long/Short-Term Preferences).

Reproduces ``SLI_RECModel._build_seq_graph`` / ``_attention_fcn`` from
``recommenders/models/deeprec/models/sequential/sli_rec.py`` (TF-1.x) as a standalone
``nn.Module`` built on :class:`SequentialBaseModel`. It fuses a long-term ASVD
attention pooling of the history with a short-term Time4LSTM branch attended against
the target item, blends the two with a per-example sigmoid gate (alpha), and
concatenates the fused user embedding with the target item embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recommenders.models.deeprec.models.sequential.pytorch.rnn_cell_pytorch import (
    Time4LSTMCell,
    time4lstm_scan,
)
from recommenders.models.deeprec.models.sequential.pytorch.sequential_base_pytorch import (
    Attention,
    FcnNet,
    SequentialBaseModel,
    init_weight_,
)

__all__ = ["SLI_RECModel"]

_MASK_PADDING = -(2**32) + 1  # matches TF's masked-softmax sentinel


class AttentionFcn(nn.Module):
    """Short-term masked attention (``_attention_fcn``) of history vs. target item."""

    def __init__(
        self,
        hidden_dim: int,
        query_dim: int,
        att_fcn_layer_sizes: list[int],
        activation: list[str],
        dropout: list[float],
        user_dropout: bool,
        enable_BN: bool,
        init_method: str,
        init_value: float,
    ) -> None:
        super().__init__()
        self.attention_mat = nn.Parameter(torch.empty(hidden_dim, query_dim))
        init_weight_(self.attention_mat, init_method, init_value)
        self.fcn = FcnNet(
            4 * query_dim,
            att_fcn_layer_sizes,
            activation,
            dropout,
            user_dropout,
            enable_BN,
            init_method,
            init_value,
        )

    def forward(
        self, query: torch.Tensor, user_embedding: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """query ``[B, E]``, user_embedding ``[B, T, H]``, mask ``[B, T]`` -> ``[B, T, H]``."""
        att_inputs = torch.einsum("bth,he->bte", user_embedding, self.attention_mat)
        t = att_inputs.shape[1]
        queries = query.unsqueeze(1).expand(-1, t, -1)
        feat = torch.cat(
            [att_inputs, queries, att_inputs - queries, att_inputs * queries], dim=-1
        )
        scores = self.fcn(feat).squeeze(-1)
        scores = torch.where(
            mask == 1.0, scores, torch.full_like(scores, _MASK_PADDING)
        )
        att_weights = torch.softmax(scores, dim=-1)
        return user_embedding * att_weights.unsqueeze(-1)


class SLI_RECModel(SequentialBaseModel):
    """SLi-Rec model (PyTorch).

    :Citation:

        Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
        Long and Short-Term Preferences for Personalized Recommendation", IJCAI'19,
        Pages 4213-4219, AAAI Press, 2019.
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
        hidden_size: int = 40,
        attention_size: int = 40,
        layer_sizes: list[int] | None = None,
        att_fcn_layer_sizes: list[int] | None = None,
        activation: list[str] | None = None,
        dropout: list[float] | None = None,
        user_dropout: bool = True,
        enable_BN: bool = True,
        init_method: str = "tnormal",
        init_value: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Build the PyTorch SLi-Rec model.

        Architecture arguments live on the constructor; training-time knobs
        (epochs, learning rate, batch size, regularization, ...) belong on
        :meth:`fit`. The user/item/category vocabulary pickles size the embedding
        tables and are also used by the internal data loader.

        Args:
            user_vocab, item_vocab, cate_vocab (str): Paths to the vocabulary pickles.
            max_seq_length (int): Maximum history length.
            item_embedding_dim, cate_embedding_dim, user_embedding_dim (int): Embedding
                sizes. ``item_embedding_dim + cate_embedding_dim`` must equal both
                ``hidden_size`` and ``attention_size`` (the alpha blend requires it).
            hidden_size (int): Time4LSTM hidden size.
            attention_size (int): Long-term ASVD attention size.
            layer_sizes (list[int]): Final prediction MLP layer sizes.
            att_fcn_layer_sizes (list[int]): Attention MLP layer sizes.
            activation (list[str]): Per-layer activations for the MLP heads.
            dropout (list[float]): Per-layer dropout rates.
            user_dropout (bool): Whether to apply dropout in the MLP heads.
            enable_BN (bool): Whether to use batch normalization in the MLP heads.
            init_method (str): Weight init scheme (``tnormal`` by default).
            init_value (float): Std/scale for weight initialization.
            seed (int): Random seed.
        """
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.att_fcn_layer_sizes = (
            att_fcn_layer_sizes if att_fcn_layer_sizes is not None else [80, 40]
        )
        super().__init__(
            user_vocab=user_vocab,
            item_vocab=item_vocab,
            cate_vocab=cate_vocab,
            max_seq_length=max_seq_length,
            item_embedding_dim=item_embedding_dim,
            cate_embedding_dim=cate_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            layer_sizes=layer_sizes,
            activation=activation,
            dropout=dropout,
            user_dropout=user_dropout,
            enable_BN=enable_BN,
            init_method=init_method,
            init_value=init_value,
            seed=seed,
        )
        self._finalize()

    def _build_seq_graph(self) -> int:
        e = self.item_embedding_dim + self.cate_embedding_dim
        hidden = self.hidden_size
        if e != hidden:
            raise ValueError(
                f"SLi-Rec requires item_embedding_dim + cate_embedding_dim == hidden_size "
                f"(alpha blend of att_fea1[{e}] and att_fea2[{hidden}]); got {e} != {hidden}."
            )
        if self.attention_size != e:
            raise ValueError(
                f"SLi-Rec requires attention_size == item+cate embedding dim; "
                f"got {self.attention_size} != {e}."
            )
        self.e_dim = e

        self.asvd_attention = Attention(
            e, self.attention_size, self.init_method, self.init_value
        )
        self.cell = Time4LSTMCell(self.item_embedding_dim, hidden)
        self.attention_fcn = AttentionFcn(
            hidden,
            e,
            self.att_fcn_layer_sizes,
            self.activation,
            self.dropout,
            self.user_dropout,
            self.enable_BN,
            self.init_method,
            self.init_value,
        )
        self.fcn_alpha = FcnNet(
            2 * e + hidden + 1,
            self.att_fcn_layer_sizes,
            self.activation,
            self.dropout,
            self.user_dropout,
            self.enable_BN,
            self.init_method,
            self.init_value,
        )
        return 2 * e

    def _seq_forward(self, batch: dict) -> torch.Tensor:
        item_hist = self.item_history_embedding
        cate_hist = self.cate_history_embedding
        mask = batch["mask"]
        seq_len = mask.sum(dim=1).long()

        # Long-term ASVD attention (unmasked).
        hist_input = torch.cat([item_hist, cate_hist], dim=2)
        att_fea1 = self.asvd_attention(hist_input).sum(dim=1)

        # Short-term Time4LSTM over [item_emb | time_from_first_action | time_to_now].
        rnn_input = torch.cat(
            [
                item_hist,
                batch["time_from_first_action"].unsqueeze(-1),
                batch["time_to_now"].unsqueeze(-1),
            ],
            dim=-1,
        )
        rnn_outputs = time4lstm_scan(self.cell, rnn_input, seq_len)
        att_fea2 = self.attention_fcn(
            self.target_item_embedding, rnn_outputs, mask
        ).sum(dim=1)

        # Alpha ensemble.
        concat_all = torch.cat(
            [
                self.target_item_embedding,
                att_fea1,
                att_fea2,
                batch["time_to_now"][:, -1:],
            ],
            dim=1,
        )
        alpha_output = torch.sigmoid(self.fcn_alpha(concat_all))
        user_embed = att_fea1 * alpha_output + att_fea2 * (1.0 - alpha_output)
        return torch.cat([user_embed, self.target_item_embedding], dim=1)
