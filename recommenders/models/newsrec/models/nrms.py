# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""NRMS (Neural News Recommendation with Multi-Head Self-Attention).

The news encoder runs multi-head self-attention over the title words and pools
them with additive attention. The user encoder does the same over the news
vectors of the clicked articles.

:Citation:

    C. Wu, F. Wu, S. Ge, T. Qi, Y. Huang and X. Xie, "Neural News Recommendation
    with Multi-Head Self-Attention", in Proceedings of the 2019 Conference on
    Empirical Methods in Natural Language Processing and the 9th International
    Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019.
"""

from __future__ import annotations

import torch

from recommenders.models.newsrec.models.base_model import NewsRecModel
from recommenders.models.newsrec.models.layers import AdditiveAttention, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(NewsRecModel):
    """NRMS model."""

    def __init__(
        self,
        word_embedding_file: str,
        word_dict_file: str,
        title_size: int = 30,
        his_size: int = 50,
        head_num: int = 20,
        head_dim: int = 20,
        attention_hidden_dim: int = 200,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """Build the NRMS model.

        Args:
            head_num, head_dim (int): Self-attention heads and size of each head.
            attention_hidden_dim (int): Hidden size of the additive attention.

        See :class:`NewsRecModel` for the other arguments.
        """
        super().__init__(
            word_embedding_file, word_dict_file, title_size, his_size, dropout, seed
        )
        word_dim = self.word_embedding.embedding_dim
        news_dim = head_num * head_dim
        self.news_attention = SelfAttention(word_dim, head_num, head_dim)
        self.news_pooling = AdditiveAttention(news_dim, attention_hidden_dim)
        self.user_attention = SelfAttention(news_dim, head_num, head_dim)
        self.user_pooling = AdditiveAttention(news_dim, attention_hidden_dim)
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def encode_news(self, news: dict, users: torch.Tensor | None) -> torch.Tensor:
        """Title words ``[N, title_size]`` -> ``[N, head_num * head_dim]``."""
        x = self.dropout(self.word_embedding(news["title"]))
        x = self.dropout(self.news_attention(x))
        return self.news_pooling(x)

    def encode_user(self, clicked: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.user_pooling(self.user_attention(clicked))
