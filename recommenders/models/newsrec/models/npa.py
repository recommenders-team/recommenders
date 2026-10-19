# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""NPA (Neural News Recommendation with Personalized Attention).

The news encoder runs a CNN over the title words and the user encoder takes the
news vectors of the clicked articles. Both pool with attention whose query comes
from the user embedding, so a news vector depends on the user it is scored for.

:Citation:

    C. Wu, F. Wu, M. An, J. Huang, Y. Huang and X. Xie, "NPA: Neural News
    Recommendation with Personalized Attention", in Proceedings of the 25th ACM
    SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.newsrec.models.base_model import NewsRecModel
from recommenders.models.newsrec.models.layers import (
    PersonalizedAttentivePooling,
    conv1d,
    dense,
)

__all__ = ["NPAModel"]


class NPAModel(NewsRecModel):
    """NPA model."""

    personalized_news = True

    def __init__(
        self,
        word_embedding_file: str,
        word_dict_file: str,
        user_dict_file: str,
        title_size: int = 10,
        his_size: int = 50,
        user_emb_dim: int = 100,
        filter_num: int = 400,
        window_size: int = 3,
        attention_hidden_dim: int = 200,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """Build the NPA model.

        Args:
            user_emb_dim (int): User embedding size.
            filter_num (int): CNN filters, the size of the news and user vectors.
            window_size (int): CNN window size.
            attention_hidden_dim (int): Size of the attention queries.

        See :class:`NewsRecModel` for the other arguments.
        """
        super().__init__(
            word_embedding_file,
            word_dict_file,
            title_size,
            his_size,
            dropout,
            seed,
            user_dict_file=user_dict_file,
        )
        # User indices start at 1, so the table keeps row 0 for unknown users.
        self.user_embedding = nn.Embedding(len(self.uid2index) + 1, user_emb_dim)
        nn.init.zeros_(self.user_embedding.weight)
        self.title_conv = conv1d(
            self.word_embedding.embedding_dim, filter_num, window_size
        )
        self.word_query = dense(user_emb_dim, attention_hidden_dim)
        self.word_pooling = PersonalizedAttentivePooling(
            filter_num, attention_hidden_dim
        )
        self.news_query = dense(user_emb_dim, attention_hidden_dim)
        self.news_pooling = PersonalizedAttentivePooling(
            filter_num, attention_hidden_dim
        )
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def encode_news(self, news: dict, users: torch.Tensor | None) -> torch.Tensor:
        """Title words ``[N, title_size]`` read by users ``[N]`` -> ``[N, filter_num]``."""
        x = self.dropout(self.word_embedding(news["title"]))
        x = self.dropout(F.relu(self.title_conv(x.transpose(1, 2))).transpose(1, 2))
        return self.word_pooling(x, self.word_query(self.user_embedding(users)))

    def encode_user(self, clicked: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.news_pooling(clicked, self.news_query(self.user_embedding(users)))
