# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""PyTorch port of the time-aware LSTM cell used by SLi-Rec.

This reproduces ``Time4LSTMCell`` from
``recommenders/models/deeprec/models/sequential/rnn_cell_implement.py`` (a TF-1.x
``RNNCell``) together with the ``tf.compat.v1.nn.dynamic_rnn`` driver that runs it
over a padded, variable-length history.

The cell modulates a vanilla LSTM's cell-state update with two "time gates" derived
from two scalar time features (``time_from_first_action`` and ``time_to_now``) that
SLi-Rec appends to each step's item embedding. Per-step input layout is
``[item_embedding (d) | time_from_first_action (1) | time_to_now (1)]`` so
``inputs[:, -1]`` is ``time_to_now`` (feeds the input gate) and ``inputs[:, -2]`` is
``time_from_first_action`` (feeds the forget gate).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["Time4LSTMCell", "time4lstm_scan"]


class Time4LSTMCell(nn.Module):
    r"""Time-aware LSTM cell (SLi-Rec).

    Gate equations (``use_peepholes=False``, ``num_proj=None``, ``cell_clip=None`` —
    the configuration SLi-Rec uses)::

        x        = inputs[:, :d]            # item embedding
        t_last   = inputs[:, d:d+1]         # time_from_first_action  (scalar)
        t_now    = inputs[:, d+1:d+2]       # time_to_now             (scalar)

        t_now_in  = tanh(t_now  * time_input_w1 + time_input_bias1)   # [B, H]
        t_last_in = tanh(t_last * time_input_w2 + time_input_bias2)   # [B, H]

        t_now_state  = x @ time_kernel_w1 + t_now_in  @ time_kernel_t1 + time_bias1
        t_last_state = x @ time_kernel_w2 + t_last_in @ time_kernel_t2 + time_bias2

        i, j, f, o = split(concat([x, m_prev]) @ W_lstm + b_lstm, 4)  # order i,j,f,o
        o = o + t_now_in @ o_kernel_t1 + t_last_in @ o_kernel_t2

        c = sigmoid(f + forget_bias) * sigmoid(t_last_state) * c_prev
            + sigmoid(i) * sigmoid(t_now_state) * tanh(j)
        m = sigmoid(o) * tanh(c)

    All ``@`` are ``x @ kernel`` with kernels stored ``[in, out]`` to mirror TF's
    ``math_ops.matmul``, so the trained TF variables can be loaded without transpose
    confusion. In TF these cell variables use the default ``glorot_uniform``
    initializer (the ``initializer`` argument is ``None`` in SLi-Rec); the LSTM bias
    is zero and the ``forget_bias=1.0`` is added at runtime inside the sigmoid.
    """

    def __init__(
        self, input_size: int, num_units: int, forget_bias: float = 1.0
    ) -> None:
        """Build the cell.

        Args:
            input_size (int): Item-embedding dimension ``d`` (the RNN input is the item
                embedding only, NOT concatenated with the category embedding).
            num_units (int): Hidden size ``H``.
            forget_bias (float): Added to the forget gate pre-activation (TF default 1.0).
        """
        super().__init__()
        self.input_size = input_size
        self.num_units = num_units
        self.forget_bias = forget_bias

        d, h = input_size, num_units

        # Vanilla-LSTM gate projection over concat([x, m_prev]) -> [i, j, f, o].
        self.W_lstm = nn.Parameter(torch.empty(d + h, 4 * h))
        self.b_lstm = nn.Parameter(torch.zeros(4 * h))

        # Per-unit affine of the scalar time value (broadcast [B,1] * [H]).
        self.time_input_w1 = nn.Parameter(torch.empty(h))
        self.time_input_bias1 = nn.Parameter(torch.empty(h))
        self.time_input_w2 = nn.Parameter(torch.empty(h))
        self.time_input_bias2 = nn.Parameter(torch.empty(h))

        # Time-gate pre-activation kernels.
        self.time_kernel_w1 = nn.Parameter(torch.empty(d, h))
        self.time_kernel_t1 = nn.Parameter(torch.empty(h, h))
        self.time_bias1 = nn.Parameter(torch.empty(h))
        self.time_kernel_w2 = nn.Parameter(torch.empty(d, h))
        self.time_kernel_t2 = nn.Parameter(torch.empty(h, h))
        self.time_bias2 = nn.Parameter(torch.empty(h))

        # Output-gate time contribution (no bias).
        self.o_kernel_t1 = nn.Parameter(torch.empty(h, h))
        self.o_kernel_t2 = nn.Parameter(torch.empty(h, h))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights to match TF's default ``glorot_uniform``.

        TF's ``get_variable`` with ``initializer=None`` falls back to
        ``glorot_uniform_initializer`` for every float variable. For 1-D vectors of
        shape ``[H]`` TF computes ``fan_in = fan_out = H`` so the uniform limit is
        ``sqrt(6 / (H + H)) = sqrt(3 / H)``. Biases inside the gate follow the same
        rule (they are ``get_variable``, not zero-init) except the LSTM ``b_lstm``
        which TF zero-inits.
        """
        for p in (
            self.W_lstm,
            self.time_kernel_w1,
            self.time_kernel_t1,
            self.time_kernel_w2,
            self.time_kernel_t2,
            self.o_kernel_t1,
            self.o_kernel_t2,
        ):
            nn.init.xavier_uniform_(p)

        limit = math.sqrt(3.0 / self.num_units)  # glorot_uniform with fan_in=fan_out=H
        for p in (
            self.time_input_w1,
            self.time_input_bias1,
            self.time_input_w2,
            self.time_input_bias2,
            self.time_bias1,
            self.time_bias2,
        ):
            nn.init.uniform_(p, -limit, limit)

        nn.init.zeros_(self.b_lstm)

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one step.

        Args:
            inputs (torch.Tensor): ``[B, d + 2]`` = ``[item_emb | t_last | t_now]``.
            state (tuple): ``(c_prev, m_prev)``, each ``[B, H]``.

        Returns:
            tuple: ``(m, (c, m))`` with ``m`` the cell output ``[B, H]``.
        """
        c_prev, m_prev = state
        d = self.input_size

        x = inputs[:, :d]
        t_last_score = inputs[:, d : d + 1]
        t_now_score = inputs[:, d + 1 : d + 2]

        t_now_input = torch.tanh(t_now_score * self.time_input_w1 + self.time_input_bias1)
        t_last_input = torch.tanh(
            t_last_score * self.time_input_w2 + self.time_input_bias2
        )

        t_now_state = (
            x @ self.time_kernel_w1 + t_now_input @ self.time_kernel_t1 + self.time_bias1
        )
        t_last_state = (
            x @ self.time_kernel_w2
            + t_last_input @ self.time_kernel_t2
            + self.time_bias2
        )

        lstm_matrix = torch.cat([x, m_prev], dim=1) @ self.W_lstm + self.b_lstm
        i, j, f, o = torch.split(lstm_matrix, self.num_units, dim=1)

        o = o + t_now_input @ self.o_kernel_t1 + t_last_input @ self.o_kernel_t2

        c = torch.sigmoid(f + self.forget_bias) * torch.sigmoid(
            t_last_state
        ) * c_prev + torch.sigmoid(i) * torch.sigmoid(t_now_state) * torch.tanh(j)
        m = torch.sigmoid(o) * torch.tanh(c)

        return m, (c, m)


def time4lstm_scan(
    cell: Time4LSTMCell,
    inputs: torch.Tensor,
    sequence_length: torch.Tensor,
) -> torch.Tensor:
    """Batch-major dynamic unroll matching ``tf.compat.v1.nn.dynamic_rnn``.

    For steps ``t >= sequence_length[b]`` the output is exactly zero and the carried
    state is frozen (copied forward), reproducing ``dynamic_rnn``'s padding semantics.

    Args:
        cell (Time4LSTMCell): The cell to run.
        inputs (torch.Tensor): ``[B, T, d + 2]``.
        sequence_length (torch.Tensor): ``[B]`` int, number of valid steps per row.

    Returns:
        torch.Tensor: ``rnn_outputs`` of shape ``[B, T, H]`` (zeros past ``seq_len``).
    """
    b, t, _ = inputs.shape
    h = cell.num_units
    device = inputs.device

    c = torch.zeros(b, h, device=device, dtype=inputs.dtype)
    m = torch.zeros(b, h, device=device, dtype=inputs.dtype)
    seq_len = sequence_length.to(device).view(b, 1)

    outputs = []
    for step in range(t):
        m_t, (c_t, _) = cell(inputs[:, step, :], (c, m))
        valid = (step < seq_len).to(inputs.dtype)  # [B, 1]
        m = valid * m_t + (1.0 - valid) * m
        c = valid * c_t + (1.0 - valid) * c
        outputs.append(valid * m_t)  # zero output on padded steps

    return torch.stack(outputs, dim=1)
