# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""PyTorch port of Caser (Convolutional Sequence Embedding Recommendation).

Reproduces ``_build_seq_graph`` / ``_add_cnn`` from the TF-1.x ``CaserModel`` in
``recommenders/models/deeprec/models/sequential/caser.py`` as a standalone
``nn.Module`` built on :class:`SequentialBaseModel`. The history embeddings of the
items and of the categories are each passed through a *vertical* convolution (one
filter spanning the whole embedding axis) and ``L`` *horizontal* convolutions (filter
heights ``1..L`` along the time axis, followed by max pooling over time). The pooled
features of both parts are concatenated with the target item embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recommenders.models.deeprec.models.sequential.pytorch.sequential_base import (
    SequentialBaseModel,
)

__all__ = ["CaserModel"]


class CaserConv(nn.Module):
    """Vertical + horizontal convolutions over one history embedding matrix.

    Mirrors the TF ``_add_cnn``: ``conv1d`` layers with ``valid`` padding and ReLU,
    Glorot-uniform kernels and zero biases (the TF ``conv1d`` defaults).
    """

    def __init__(
        self, max_seq_length: int, embedding_dim: int, n_v: int, n_h: int, L: int
    ) -> None:
        super().__init__()
        # Vertical: TF convolves the transposed ``[B, E, T]`` input, i.e. the filter
        # slides over the embedding axis with ``T`` input channels. In PyTorch's
        # ``[B, C, L]`` layout this is ``[B, T, E]`` with kernel size ``E``.
        self.vertical = nn.Conv1d(max_seq_length, n_v, kernel_size=embedding_dim)
        # Horizontal: filters of height ``h`` slide over time with ``E`` channels.
        self.horizontal = nn.ModuleList(
            [nn.Conv1d(embedding_dim, n_h, kernel_size=h) for h in range(1, L + 1)]
        )
        for conv in [self.vertical, *self.horizontal]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, hist_matrix: torch.Tensor) -> torch.Tensor:
        """hist_matrix ``[B, T, E]`` -> ``[B, n_v + L * n_h]``."""
        out_v = torch.relu(self.vertical(hist_matrix)).flatten(start_dim=1)
        hist_t = hist_matrix.transpose(1, 2)  # [B, E, T]
        out_hs = [torch.relu(conv(hist_t)).amax(dim=-1) for conv in self.horizontal]
        return torch.cat([out_v, *out_hs], dim=1)


class CaserModel(SequentialBaseModel):
    """Caser model (PyTorch).

    :Citation:

        J. Tang and K. Wang, "Personalized top-n sequential recommendation via convolutional
        sequence embedding", in Proceedings of the Eleventh ACM International Conference on
        Web Search and Data Mining, ACM, 2018.
    """

    def __init__(
        self,
        user_vocab: str,
        item_vocab: str,
        cate_vocab: str,
        max_seq_length: int = 50,
        min_seq_length: int = 5,
        item_embedding_dim: int = 32,
        cate_embedding_dim: int = 8,
        user_embedding_dim: int = 16,
        L: int = 3,
        n_v: int = 128,
        n_h: int = 128,
        layer_sizes: list[int] | None = None,
        activation: list[str] | None = None,
        dropout: list[float] | None = None,
        user_dropout: bool = True,
        enable_BN: bool = True,
        init_method: str = "tnormal",
        init_value: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Build the PyTorch Caser model.

        Architecture arguments live on the constructor; training-time knobs
        (epochs, learning rate, batch size, regularization, ...) belong on
        :meth:`fit`. The user/item/category vocabulary pickles size the embedding
        tables and are also used by the internal data loader.

        Args:
            user_vocab, item_vocab, cate_vocab (str): Paths to the vocabulary pickles.
            max_seq_length (int): Maximum history length. It is the number of input
                channels of the vertical convolution.
            min_seq_length (int): Minimum history length of an instance to be used.
            item_embedding_dim, cate_embedding_dim, user_embedding_dim (int): Embedding
                sizes.
            L (int): Number of horizontal convolutions, with filter heights ``1..L``.
                It must not exceed ``max_seq_length``.
            n_v (int): Number of vertical convolution filters.
            n_h (int): Number of filters of each horizontal convolution.
            layer_sizes (list[int]): Final prediction MLP layer sizes.
            activation (list[str]): Per-layer activations for the MLP head.
            dropout (list[float]): Per-layer dropout rates.
            user_dropout (bool): Whether to apply dropout in the MLP head.
            enable_BN (bool): Whether to use batch normalization in the MLP head.
            init_method (str): Weight init scheme (``tnormal`` by default). It applies
                to the embeddings and the MLP head; the convolutions use Glorot
                uniform as in the TF implementation.
            init_value (float): Std/scale for weight initialization.
            seed (int): Random seed.
        """
        if L < 1 or L > max_seq_length:
            raise ValueError(
                f"Caser requires 1 <= L <= max_seq_length; got L={L}, "
                f"max_seq_length={max_seq_length}."
            )
        self.L = L
        self.n_v = n_v
        self.n_h = n_h
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
        self.min_seq_length = min_seq_length
        self._finalize()

    def _build_seq_graph(self) -> int:
        self.item_conv = CaserConv(
            self.max_seq_length, self.item_embedding_dim, self.n_v, self.n_h, self.L
        )
        self.cate_conv = CaserConv(
            self.max_seq_length, self.cate_embedding_dim, self.n_v, self.n_h, self.L
        )
        cnn_dim = 2 * (self.n_v + self.L * self.n_h)
        return cnn_dim + self.item_embedding_dim + self.cate_embedding_dim

    def _seq_forward(self, batch: dict) -> torch.Tensor:
        item_out = self.item_conv(self.item_history_embedding)
        cate_out = self.cate_conv(self.cate_history_embedding)
        return torch.cat([item_out, cate_out, self.target_item_embedding], dim=1)
