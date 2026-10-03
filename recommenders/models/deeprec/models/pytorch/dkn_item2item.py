# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""DKN adapted to item-to-item recommendation.

The same KCNN embeds every article, a ``tanh`` layer densifies the embedding and an
L2 normalization puts it on the unit sphere, so the relation score of two
articles is their cosine similarity. Training contrasts each source article's
related target with ``neg_num`` unrelated ones through a softmax.

The tutorial is
``examples/07_tutorials/KDD2020-tutorial/step4_run_dkn_item2item.ipynb``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.deeprec.io.dkn_dataset import DKNItem2ItemDataset
from recommenders.models.deeprec.models.pytorch.dkn import (
    DKNBase,
    glorot_truncated_normal_,
)

__all__ = ["DKNItem2Item"]


class DKNItem2Item(DKNBase):
    """DKN for item-to-item recommendations."""

    def __init__(
        self,
        news_feature_file: str,
        word_embedding_file: str,
        neg_num: int,
        entity_embedding_file: str | None = None,
        context_embedding_file: str | None = None,
        filter_sizes: list[int] | None = None,
        num_filters: int = 100,
        init_method: str = "uniform",
        init_value: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Build the item-to-item DKN model.

        Architecture arguments live on the constructor; training-time knobs belong
        on :meth:`fit`. The arguments shared with
        :class:`~recommenders.models.deeprec.models.pytorch.dkn.DKN` mean the same.

        Args:
            neg_num (int): Unrelated articles per training group, which the data
                files hold as ``neg_num + 2`` consecutive lines: the source, its
                related target, then the unrelated ones.
            init_method (str): ``uniform`` or ``xavier_normal``, the init of the
                convolution biases.
        """
        super().__init__(
            word_embedding_file,
            entity_embedding_file,
            context_embedding_file,
            filter_sizes,
            num_filters,
            init_method,
            init_value,
            seed,
        )
        self.iterator = DKNItem2ItemDataset(news_feature_file, neg_num)

        news_dim = self.kcnn.output_dim
        self.doc_transform = nn.Linear(news_dim, news_dim, bias=False)
        glorot_truncated_normal_(self.doc_transform.weight, news_dim)
        self.layer_params.append(self.doc_transform.weight)

        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def _embed_news(self, words: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        """Unit-norm article embeddings ``[N, output_dim]``."""
        news = torch.tanh(self.doc_transform(self.kcnn(words, entities)))
        # The epsilon bounds the squared norm, so a zero vector stays zero.
        return news * torch.rsqrt(
            news.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Relation scores ``[B, neg_num + 1]`` of each source to its targets."""
        batch_size, group_size, doc_size = batch["words"].shape
        news = self._embed_news(
            batch["words"].reshape(-1, doc_size),
            batch["entities"].reshape(-1, doc_size),
        ).view(batch_size, group_size, -1)
        return (news[:, 1:] * news[:, :1]).sum(dim=-1)

    def _data_loss(self, batch: dict) -> torch.Tensor:
        """Negative log softmax probability of the related target, summed over groups."""
        pred = torch.softmax(self.forward(batch), dim=-1)
        return -torch.log(pred[:, 0] + 1e-10).sum()

    @torch.no_grad()
    def run_eval(
        self,
        filename: str,
        batch_size: int = 100,
        metrics: list[str] | None = None,
        pairwise_metrics: list[str] | None = None,
    ) -> dict:
        """Evaluate ``filename`` and return the metric dictionary.

        Every group is scored with the softmax over its targets; the related target
        is the positive and the unrelated ones the negatives.

        Args:
            filename (str): A file name that will be evaluated.
            batch_size (int): Groups per mini-batch.
            metrics (list[str]): Metrics over all the target scores. Defaults to
                none.
            pairwise_metrics (list[str]): Metrics averaged over the groups.
                Defaults to ``["group_auc", "mean_mrr", "ndcg@5;10"]``.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        metrics = metrics if metrics is not None else []
        pairwise_metrics = (
            pairwise_metrics
            if pairwise_metrics is not None
            else ["group_auc", "mean_mrr", "ndcg@5;10"]
        )
        self.eval()
        preds = []
        for np_batch, _ in self.iterator.load_data_from_file(filename, batch_size):
            pred = torch.softmax(self.forward(self._to_tensors(np_batch)), dim=-1)
            preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.zeros_like(preds, dtype=np.int32)
        labels[:, 0] = 1

        res = cal_metric(labels.reshape(-1), preds.reshape(-1), metrics)
        res.update(cal_metric(labels, preds, pairwise_metrics))
        return res
