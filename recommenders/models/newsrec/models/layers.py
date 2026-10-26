# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Attention layers shared by the news recommendation models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdditiveAttention",
    "SelfAttention",
    "PersonalizedAttentivePooling",
    "dense",
    "conv1d",
]


def dense(in_dim: int, out_dim: int, bias: bool = True) -> nn.Linear:
    """Linear layer with a Glorot uniform weight and a zero bias."""
    linear = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(linear.weight)
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


def conv1d(in_dim: int, filter_num: int, window_size: int) -> nn.Conv1d:
    """Length-preserving convolution with a Glorot uniform kernel and a zero bias."""
    conv = nn.Conv1d(in_dim, filter_num, window_size, padding="same")
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class AdditiveAttention(nn.Module):
    """Pool ``[N, L, D]`` into ``[N, D]`` with a learned query vector."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = dense(in_dim, hidden_dim)
        self.query = dense(hidden_dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Weighted sum of ``x`` over its second axis.

        Args:
            mask (torch.Tensor): ``[N, L]`` boolean, False for the positions left
                out. A row with every position left out pools to zeros.
        """
        scores = self.query(torch.tanh(self.projection(x))).squeeze(-1)
        # The shift keeps exp finite; it is detached because it cancels out.
        weights = torch.exp(scores - scores.amax(dim=1, keepdim=True).detach())
        if mask is not None:
            weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-7)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, without an output projection."""

    def __init__(self, in_dim: int, head_num: int, head_dim: int) -> None:
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.query = dense(in_dim, head_num * head_dim, bias=False)
        self.key = dense(in_dim, head_num * head_dim, bias=False)
        self.value = dense(in_dim, head_num * head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, L, in_dim]`` -> ``[N, L, head_num * head_dim]``."""

        def heads(linear: nn.Linear) -> torch.Tensor:
            return (
                linear(x)
                .view(*x.shape[:2], self.head_num, self.head_dim)
                .transpose(1, 2)
            )

        out = F.scaled_dot_product_attention(
            heads(self.query), heads(self.key), heads(self.value)
        )
        return out.transpose(1, 2).flatten(2)


class PersonalizedAttentivePooling(nn.Module):
    """Pool ``[N, L, D]`` into ``[N, D]`` with a query given per row."""

    def __init__(self, in_dim: int, query_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = dense(in_dim, query_dim)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Weighted sum of ``x`` with attention from ``query`` ``[N, query_dim]``."""
        x = self.dropout(x)
        scores = torch.einsum("nlq,nq->nl", torch.tanh(self.projection(x)), query)
        return (x * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)
