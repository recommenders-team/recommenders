# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Shared PyTorch building blocks for the deeprec models.

Port of the reusable pieces of the TF-1.x ``base_model.py``: the weight
initializers (``_get_initializer``), the activation table (``_activate``), and the
``FcnNet`` MLP head (``_fcn_net`` / ``_build_dnn``, which are the same network).

Numeric conventions preserved from TF:

* ``tnormal`` init maps to ``trunc_normal_(std=init_value)``; biases are zero.
* BatchNorm: TF ``momentum=0.95`` -> PyTorch ``momentum=0.05``; ``eps=1e-4``.
* Dropout is applied BEFORE the activation, only on hidden layers, and only when
  ``user_dropout`` is set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "relu": F.relu,
    "tanh": torch.tanh,
    "elu": F.elu,
    "identity": lambda x: x,
}


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
        self.acts = [ACTIVATIONS[a] for a in activation]

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
