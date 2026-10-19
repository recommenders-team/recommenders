# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""LSTUR (Neural News Recommendation with Long- and Short-term User Representations).

The news encoder runs a CNN over the title words and pools the non-padding ones
with additive attention. A GRU over the clicked articles gives the short-term user
representation and a user embedding the long-term one. ``ini`` starts the GRU from
the long-term representation; ``con`` concatenates both and projects them back.

:Citation:

    M. An, F. Wu, C. Wu, K. Zhang, Z. Liu and X. Xie, "Neural News Recommendation
    with Long- and Short-term User Representations", in Proceedings of the 57th
    Annual Meeting of the Association for Computational Linguistics, ACL 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.newsrec.models.base_model import NewsRecModel
from recommenders.models.newsrec.models.layers import AdditiveAttention, conv1d, dense

__all__ = ["LSTURModel"]


class LSTURModel(NewsRecModel):
    """LSTUR model."""

    def __init__(
        self,
        word_embedding_file: str,
        word_dict_file: str,
        user_dict_file: str,
        title_size: int = 30,
        his_size: int = 50,
        filter_num: int = 400,
        window_size: int = 3,
        combine: str = "ini",
        attention_hidden_dim: int = 200,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """Build the LSTUR model.

        Args:
            filter_num (int): CNN filters, also the GRU and user embedding size,
                since the news and user vectors meet in a dot product.
            window_size (int): CNN window size.
            combine (str): ``ini`` or ``con``, how the long- and short-term user
                representations are combined.
            attention_hidden_dim (int): Hidden size of the additive attention.

        See :class:`NewsRecModel` for the other arguments.
        """
        if combine not in ("ini", "con"):
            raise ValueError(
                "combine must be ini or con, but now is {0}".format(combine)
            )
        super().__init__(
            word_embedding_file,
            word_dict_file,
            title_size,
            his_size,
            dropout,
            seed,
            user_dict_file=user_dict_file,
        )
        self.combine = combine
        self.title_conv = conv1d(
            self.word_embedding.embedding_dim, filter_num, window_size
        )
        self.title_pooling = AdditiveAttention(filter_num, attention_hidden_dim)
        # User indices start at 1, so the table keeps row 0 for unknown users.
        self.user_embedding = nn.Embedding(len(self.uid2index) + 1, filter_num)
        nn.init.zeros_(self.user_embedding.weight)
        self.gru = nn.GRUCell(filter_num, filter_num)
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.xavier_uniform_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)
        if combine == "con":
            self.combine_dense = dense(2 * filter_num, filter_num)
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def encode_news(self, news: dict, users: torch.Tensor | None) -> torch.Tensor:
        """Title words ``[N, title_size]`` -> news vectors ``[N, filter_num]``.

        An article without a known title word gets a zero vector.
        """
        words = news["title"]
        x = self.dropout(self.word_embedding(words))
        x = self.dropout(F.relu(self.title_conv(x.transpose(1, 2))).transpose(1, 2))
        return self.title_pooling(x, mask=words != 0)

    def encode_user(self, clicked: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        """Run the GRU over the clicked articles, skipping the zero news vectors."""
        long_term = self.user_embedding(users)
        state = long_term if self.combine == "ini" else torch.zeros_like(long_term)
        steps = clicked.ne(0).any(dim=-1)
        for position in range(clicked.shape[1]):
            state = torch.where(
                steps[:, position : position + 1],
                self.gru(clicked[:, position], state),
                state,
            )
        if self.combine == "ini":
            return state
        return self.combine_dense(torch.cat([state, long_term], dim=-1))
