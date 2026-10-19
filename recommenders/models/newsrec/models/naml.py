# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""NAML (Neural News Recommendation with Attentive Multi-View Learning).

The news encoder embeds four views of an article: the title and the abstract with
a CNN and additive attention, the category and the subcategory with an embedding
and a dense layer. Additive attention pools the views into the news vector, and
the user encoder pools the news vectors of the clicked articles.

:Citation:

    C. Wu, F. Wu, M. An, J. Huang, Y. Huang and X. Xie, "Neural News
    Recommendation with Attentive Multi-View Learning", in Proceedings of the
    Twenty-Eighth International Joint Conference on Artificial Intelligence,
    IJCAI 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.newsrec.models.base_model import NewsRecModel
from recommenders.models.newsrec.models.layers import AdditiveAttention, conv1d, dense

__all__ = ["NAMLModel"]


def _category_embedding(count: int, dim: int) -> nn.Embedding:
    embedding = nn.Embedding(count, dim)
    nn.init.uniform_(embedding.weight, -0.05, 0.05)
    return embedding


class NAMLModel(NewsRecModel):
    """NAML model."""

    def __init__(
        self,
        word_embedding_file: str,
        word_dict_file: str,
        vert_dict_file: str,
        subvert_dict_file: str,
        title_size: int = 30,
        body_size: int = 50,
        his_size: int = 50,
        vert_emb_dim: int = 100,
        subvert_emb_dim: int = 100,
        filter_num: int = 400,
        window_size: int = 3,
        attention_hidden_dim: int = 200,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """Build the NAML model.

        Args:
            vert_emb_dim, subvert_emb_dim (int): Category and subcategory embedding
                sizes.
            filter_num (int): CNN filters, the size of the news and user vectors.
            window_size (int): CNN window size.
            attention_hidden_dim (int): Hidden size of the additive attention.

        See :class:`NewsRecModel` for the other arguments.
        """
        super().__init__(
            word_embedding_file,
            word_dict_file,
            title_size,
            his_size,
            dropout,
            seed,
            body_size=body_size,
            vert_dict_file=vert_dict_file,
            subvert_dict_file=subvert_dict_file,
        )
        word_dim = self.word_embedding.embedding_dim
        self.title_conv = conv1d(word_dim, filter_num, window_size)
        self.title_pooling = AdditiveAttention(filter_num, attention_hidden_dim)
        self.body_conv = conv1d(word_dim, filter_num, window_size)
        self.body_pooling = AdditiveAttention(filter_num, attention_hidden_dim)
        # Category indices start at 1, so the tables keep row 0 for unknown ones.
        self.vert_embedding = _category_embedding(len(self.vert_dict) + 1, vert_emb_dim)
        self.vert_dense = dense(vert_emb_dim, filter_num)
        self.subvert_embedding = _category_embedding(
            len(self.subvert_dict) + 1, subvert_emb_dim
        )
        self.subvert_dense = dense(subvert_emb_dim, filter_num)
        self.view_pooling = AdditiveAttention(filter_num, attention_hidden_dim)
        self.user_pooling = AdditiveAttention(filter_num, attention_hidden_dim)
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def _encode_text(
        self, words: torch.Tensor, conv: nn.Conv1d, pooling: AdditiveAttention
    ) -> torch.Tensor:
        x = self.dropout(self.word_embedding(words))
        x = self.dropout(F.relu(conv(x.transpose(1, 2))).transpose(1, 2))
        return pooling(x)

    def encode_news(self, news: dict, users: torch.Tensor | None) -> torch.Tensor:
        """Title, body, vert and subvert of ``N`` articles -> ``[N, filter_num]``."""
        views = [
            self._encode_text(news["title"], self.title_conv, self.title_pooling),
            self._encode_text(news["body"], self.body_conv, self.body_pooling),
            F.relu(self.vert_dense(self.vert_embedding(news["vert"]))),
            F.relu(self.subvert_dense(self.subvert_embedding(news["subvert"]))),
        ]
        return self.view_pooling(torch.stack(views, dim=1))

    def encode_user(self, clicked: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.user_pooling(clicked)
