# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Training and evaluation shared by the news recommendation models.

A model scores a candidate article by the dot product of its news vector with the
user vector. It is trained on samples of one clicked and ``npratio`` non-clicked
candidates with a softmax cross-entropy loss, and evaluated by ranking the
candidates of every impression.
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.newsrec.io.mind_dataset import MINDDataset

__all__ = ["NewsRecModel"]


def _load_dict(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)


class NewsRecModel(nn.Module):
    """Base of NRMS, NAML, LSTUR and NPA.

    A subclass builds its layers after calling ``__init__`` and implements
    ``encode_news(news, users)``, mapping article features to news vectors, and
    ``encode_user(clicked, users)``, mapping the news vectors of the clicked
    articles ``[N, his_size, D]`` to user vectors ``[N, D]``.
    """

    # Whether encode_news depends on the user, which rules out embedding every
    # article once per evaluation.
    personalized_news = False

    def __init__(
        self,
        word_embedding_file: str,
        word_dict_file: str,
        title_size: int,
        his_size: int,
        dropout: float,
        seed: int | None,
        user_dict_file: str | None = None,
        body_size: int | None = None,
        vert_dict_file: str | None = None,
        subvert_dict_file: str | None = None,
    ) -> None:
        """Load the dictionaries and the pre-trained word embeddings.

        Args:
            word_embedding_file (str): ``.npy`` file of pre-trained
                ``[word_count + 1, word_dim]`` word embeddings, fine-tuned during
                training.
            word_dict_file (str): Pickled word to word index dictionary.
            title_size (int): Title words kept per article.
            his_size (int): Clicked articles kept per user.
            dropout (float): Dropout rate on the word embeddings and on the
                convolution or self-attention outputs.
            seed (int): Random seed of the weights and of the negative sampling.
            user_dict_file (str): Pickled user ID to user index dictionary, for the
                models with user embeddings.
            body_size (int): Abstract words kept per article, for NAML.
            vert_dict_file, subvert_dict_file (str): Pickled category and
                subcategory to index dictionaries, for NAML.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.title_size = title_size
        self.his_size = his_size
        self.body_size = body_size
        self.word_dict = _load_dict(word_dict_file)
        self.uid2index = None if user_dict_file is None else _load_dict(user_dict_file)
        self.vert_dict = None if vert_dict_file is None else _load_dict(vert_dict_file)
        self.subvert_dict = (
            None if subvert_dict_file is None else _load_dict(subvert_dict_file)
        )
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(np.load(word_embedding_file), dtype=torch.float32),
            freeze=False,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self) -> torch.device:
        """Device the model's parameters live on."""
        return self.word_embedding.weight.device

    def load_dataset(self, news_file: str, behaviors_file: str) -> MINDDataset:
        """Read a MIND split with the model's dictionaries and sizes."""
        return MINDDataset(
            news_file,
            behaviors_file,
            self.word_dict,
            self.title_size,
            self.his_size,
            uid2index=self.uid2index,
            body_size=self.body_size,
            vert_dict=self.vert_dict,
            subvert_dict=self.subvert_dict,
        )

    def _news_tensors(self, dataset: MINDDataset) -> dict:
        return {
            name: torch.as_tensor(value, device=self.device)
            for name, value in dataset.news.items()
        }

    def forward(self, news: dict, batch: dict) -> torch.Tensor:
        """Score logits ``[N, C]`` of the candidates of every sample.

        Args:
            news (dict): Article features of the split, as tensors.
            batch (dict): ``candidate_rows`` ``[N, C]``, ``clicked_rows``
                ``[N, his_size]`` and ``users`` ``[N]`` tensors.
        """
        users = batch["users"]
        candidates = self._encode_rows(news, batch["candidate_rows"], users)
        user = self.encode_user(
            self._encode_rows(news, batch["clicked_rows"], users), users
        )
        return (candidates * user.unsqueeze(1)).sum(dim=-1)

    def _encode_rows(
        self, news: dict, rows: torch.Tensor, users: torch.Tensor
    ) -> torch.Tensor:
        """News vectors ``[N, K, D]`` of the articles at ``rows`` ``[N, K]``."""
        features = {name: value[rows.flatten()] for name, value in news.items()}
        vectors = self.encode_news(features, users.repeat_interleave(rows.shape[1]))
        return vectors.view(*rows.shape, -1)

    def fit(
        self,
        train_news_file: str,
        train_behaviors_file: str,
        valid_news_file: str,
        valid_behaviors_file: str,
        test_news_file: str | None = None,
        test_behaviors_file: str | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        npratio: int = 4,
        metrics: list[str] | None = None,
        step_limit: int | None = None,
    ):
        """Train on the train split, evaluating on the valid split after every epoch.

        Args:
            test_news_file, test_behaviors_file (str): Optional test split, also
                evaluated after every epoch.
            batch_size (int): Training samples per mini-batch.
            learning_rate (float): Adam learning rate.
            npratio (int): Non-clicked candidates per training sample.
            metrics (list[str]): Passed to :meth:`run_eval`.
            step_limit (int): Training steps per epoch. ``None`` goes through the
                whole train split.

        Returns:
            object: An instance of self.
        """
        train = self.load_dataset(train_news_file, train_behaviors_file)
        evaluated = [("eval", self.load_dataset(valid_news_file, valid_behaviors_file))]
        if test_news_file is not None:
            evaluated.append(
                ("test", self.load_dataset(test_news_file, test_behaviors_file))
            )
        news = self._news_tensors(train)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(1, epochs + 1):
            train_start = time.time()
            self.train()
            epoch_loss = 0.0
            step = 0
            for np_batch in train.load_train_batches(batch_size, npratio, self.rng):
                if step_limit is not None and step >= step_limit:
                    break
                batch = {
                    name: torch.as_tensor(value, device=self.device)
                    for name, value in np_batch.items()
                }
                logits = self.forward(news, batch)
                # The clicked article is always the first candidate.
                loss = F.cross_entropy(
                    logits, logits.new_zeros(len(logits), dtype=torch.long)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1
            train_time = time.time() - train_start

            eval_start = time.time()
            message = "at epoch {0:d}\ntrain info: logloss loss:{1}".format(
                epoch, epoch_loss / step
            )
            for name, dataset in evaluated:
                res = self._evaluate(dataset, metrics)
                message += "\n{0} info: {1}".format(
                    name,
                    ", ".join(
                        "{0}:{1}".format(metric, value)
                        for metric, value in sorted(res.items())
                    ),
                )
            print(message)
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, time.time() - eval_start
                )
            )
        return self

    def run_eval(
        self,
        news_file: str,
        behaviors_file: str,
        metrics: list[str] | None = None,
        batch_size: int = 1024,
    ) -> dict:
        """Evaluate the ranking of the candidates of every impression.

        Args:
            metrics (list[str]): Metrics averaged over the impressions. Defaults to
                ``["group_auc", "mean_mrr", "ndcg@5;10"]``.
            batch_size (int): Articles or impressions per encoder pass.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        return self._evaluate(
            self.load_dataset(news_file, behaviors_file), metrics, batch_size
        )

    def _evaluate(
        self, dataset: MINDDataset, metrics: list[str] | None, batch_size: int = 1024
    ) -> dict:
        metrics = (
            metrics if metrics is not None else ["group_auc", "mean_mrr", "ndcg@5;10"]
        )
        return cal_metric(dataset.labels, self._score(dataset, batch_size), metrics)

    def predict(
        self, news_file: str, behaviors_file: str, batch_size: int = 1024
    ) -> list[np.ndarray]:
        """Score logits of the candidates of every impression, in file order.

        Args:
            batch_size (int): Articles or impressions per encoder pass.

        Returns:
            list: One array per impression, in the order of its candidates.
        """
        return self._score(self.load_dataset(news_file, behaviors_file), batch_size)

    @torch.no_grad()
    def _score(self, dataset: MINDDataset, batch_size: int) -> list[np.ndarray]:
        self.eval()
        news = self._news_tensors(dataset)
        histories = torch.as_tensor(dataset.histories, device=self.device)
        users = torch.as_tensor(dataset.users, device=self.device)
        candidates = torch.as_tensor(
            np.concatenate(dataset.impressions), device=self.device
        )
        sizes = [len(impression) for impression in dataset.impressions]
        owners = torch.repeat_interleave(
            torch.arange(len(sizes), device=self.device),
            torch.as_tensor(sizes, device=self.device),
        )
        if self.personalized_news:
            offsets = np.cumsum([0] + sizes)
            step = max(1, batch_size // self.his_size)
            scores = []
            for start in range(0, len(sizes), step):
                stop = min(start + step, len(sizes))
                chunk_users = users[start:stop]
                user_vectors = self.encode_user(
                    self._encode_rows(news, histories[start:stop], chunk_users),
                    chunk_users,
                )
                rows = candidates[offsets[start] : offsets[stop]]
                impressions = owners[offsets[start] : offsets[stop]]
                news_vectors = self.encode_news(
                    {name: value[rows] for name, value in news.items()},
                    users[impressions],
                )
                scores.append(
                    (news_vectors * user_vectors[impressions - start]).sum(dim=-1)
                )
            scores = torch.cat(scores)
        else:
            news_vectors = torch.cat(
                [
                    self.encode_news(
                        {
                            name: value[start : start + batch_size]
                            for name, value in news.items()
                        },
                        None,
                    )
                    for start in range(0, len(news["title"]), batch_size)
                ]
            )
            user_vectors = torch.cat(
                [
                    self.encode_user(
                        news_vectors[histories[start : start + batch_size]],
                        users[start : start + batch_size],
                    )
                    for start in range(0, len(histories), batch_size)
                ]
            )
            scores = (news_vectors[candidates] * user_vectors[owners]).sum(dim=-1)
        return list(np.split(scores.cpu().numpy(), np.cumsum(sizes)[:-1]))
