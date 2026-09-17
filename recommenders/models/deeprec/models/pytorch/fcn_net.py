# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Fully-connected scoring head shared by the deeprec models.

xDeepFM and DKN both reduce a flat feature vector to a single logit with the same
network: a stack of hidden layers, each ``Linear -> [BatchNorm] -> [Dropout] ->
activation``, followed by a bare ``Linear(-, 1)``. They differ in the activation
(ReLU for xDeepFM, sigmoid for the DKN scorer, ReLU again for its attention head)
and in how the weights are initialized, so both are arguments rather than choices
baked into the class.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

__all__ = ["FcnNet"]


class FcnNet(nn.Module):
    """Multi-layer perceptron that scores a feature vector with a single logit."""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        activation: nn.Module,
        dropout: list[float],
        enable_BN: bool,
        init_weight: Callable[[torch.Tensor], None],
    ) -> None:
        """Initialize parameters.

        Args:
            input_dim (int): Size of the input feature vector.
            layer_sizes (list[int]): Hidden layer sizes.
            activation (torch.nn.Module): Activation applied after every hidden
                layer, for example ``torch.nn.ReLU()``.
            dropout (list[float]): Dropout rate per hidden layer, one per entry of
                ``layer_sizes``. A rate of ``0.0`` disables dropout on that layer.
            enable_BN (bool): Whether to insert batch normalization after every
                hidden layer.
            init_weight (Callable): Called on each ``Linear`` weight to initialize
                it in place. Biases are always zeroed.
        """
        super().__init__()
        if len(dropout) != len(layer_sizes):
            raise ValueError(
                "dropout must hold one rate per hidden layer; got {0} rates for "
                "{1} layers.".format(len(dropout), len(layer_sizes))
            )
        self.activation = activation

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        last = input_dim
        for size, rate in zip(layer_sizes, dropout):
            lin = nn.Linear(last, size)
            init_weight(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)
            self.bns.append(
                nn.BatchNorm1d(size, momentum=0.05, eps=1e-4)
                if enable_BN
                else nn.Identity()
            )
            self.dropouts.append(nn.Dropout(p=rate))
            last = size

        self.out = nn.Linear(last, 1)
        init_weight(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape ``[B, input_dim]``.

        Returns:
            torch.Tensor: Logit of shape ``[B, 1]``.
        """
        for lin, bn, drop in zip(self.linears, self.bns, self.dropouts):
            x = self.activation(drop(bn(lin(x))))
        return self.out(x)
