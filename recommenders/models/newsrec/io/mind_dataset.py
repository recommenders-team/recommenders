# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Loader of one MIND split for the news recommendation models.

The articles of a split are read into arrays with one row per article. Row 0 is
an all-zero padding article that fills short user histories and short negative
samples, so every other array holds article rows instead of article features.
"""

from __future__ import annotations

import numpy as np

from recommenders.models.newsrec.newsrec_utils import word_tokenize

__all__ = ["MINDDataset"]


def _word_indices(text: str, word_dict: dict, size: int) -> list[int]:
    """Index of each of the first ``size`` words, 0 for an unknown word or padding."""
    words = word_tokenize(text)[:size]
    return [word_dict.get(word, 0) for word in words] + [0] * (size - len(words))


class MINDDataset:
    """Articles and impressions of one MIND split.

    Attributes:
        news (dict): ``title`` ``[news_count + 1, title_size]`` word indices, plus
            ``body`` ``[news_count + 1, body_size]`` word indices and ``vert`` and
            ``subvert`` ``[news_count + 1]`` category indices when ``body_size``
            is set.
        histories (numpy.ndarray): ``[impression_count, his_size]`` rows of the
            last clicked articles of each impression, left-padded with row 0.
        users (numpy.ndarray): ``[impression_count]`` user indices, 0 for a user
            missing from the user dictionary.
        impressions (list): Rows of the candidate articles of each impression.
        labels (list): Click labels of the candidate articles of each impression.
    """

    def __init__(
        self,
        news_file: str,
        behaviors_file: str,
        word_dict: dict,
        title_size: int,
        his_size: int,
        uid2index: dict | None = None,
        body_size: int | None = None,
        vert_dict: dict | None = None,
        subvert_dict: dict | None = None,
    ) -> None:
        """Load the articles and the impressions.

        Args:
            news_file (str): MIND ``news.tsv``.
            behaviors_file (str): MIND ``behaviors.tsv``.
            word_dict (dict): Word to word index, starting at 1.
            uid2index (dict): User ID to user index, starting at 1. ``None`` gives
                every user index 0.
            body_size (int): Abstract words kept per article. ``None`` reads only
                the titles; otherwise ``vert_dict`` and ``subvert_dict`` are
                required.
        """
        nid2row = {}
        title, body, vert, subvert = (
            [[0] * title_size],
            [[0] * (body_size or 0)],
            [0],
            [0],
        )
        with open(news_file, "r", encoding="utf-8") as rd:
            for line in rd:
                nid, category, subcategory, title_text, abstract = line.strip(
                    "\n"
                ).split("\t")[:5]
                if nid in nid2row:
                    continue
                nid2row[nid] = len(nid2row) + 1
                title.append(_word_indices(title_text, word_dict, title_size))
                if body_size is not None:
                    body.append(_word_indices(abstract, word_dict, body_size))
                    vert.append(vert_dict.get(category, 0))
                    subvert.append(subvert_dict.get(subcategory, 0))
        self.news = {"title": np.asarray(title, dtype=np.int64)}
        if body_size is not None:
            self.news["body"] = np.asarray(body, dtype=np.int64)
            self.news["vert"] = np.asarray(vert, dtype=np.int64)
            self.news["subvert"] = np.asarray(subvert, dtype=np.int64)

        histories, users = [], []
        self.impressions, self.labels = [], []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            for line in rd:
                uid, _, history, impression = line.strip("\n").split("\t")[-4:]
                history = [nid2row[nid] for nid in history.split()][-his_size:]
                histories.append([0] * (his_size - len(history)) + history)
                users.append(0 if uid2index is None else uid2index.get(uid, 0))
                candidates = [item.split("-") for item in impression.split()]
                self.impressions.append(
                    np.asarray([nid2row[nid] for nid, _ in candidates], dtype=np.int64)
                )
                self.labels.append(
                    np.asarray([int(label) for _, label in candidates], dtype=np.int64)
                )
        self.histories = np.asarray(histories, dtype=np.int64).reshape(-1, his_size)
        self.users = np.asarray(users, dtype=np.int64)

    def load_train_batches(
        self, batch_size: int, npratio: int, rng: np.random.Generator
    ):
        """Yield training samples in shuffled impression order.

        Every clicked candidate of an impression makes one sample, completed with
        ``npratio`` candidates sampled without replacement from the impression's
        non-clicked ones, padded with row 0 when there are fewer.

        Yields:
            dict: ``candidate_rows`` ``[batch_size, npratio + 1]`` with the clicked
            article first, ``clicked_rows`` ``[batch_size, his_size]`` and ``users``
            ``[batch_size]``. The last batch may hold fewer samples.
        """
        candidates, samples = [], []
        for index in rng.permutation(len(self.impressions)):
            labels = self.labels[index]
            negatives = self.impressions[index][labels == 0]
            for positive in self.impressions[index][labels == 1]:
                if len(negatives) < npratio:
                    sampled = np.concatenate(
                        [negatives, np.zeros(npratio - len(negatives), dtype=np.int64)]
                    )
                else:
                    sampled = rng.choice(negatives, npratio, replace=False)
                candidates.append(np.concatenate([[positive], sampled]))
                samples.append(index)
                if len(samples) == batch_size:
                    yield self._train_batch(candidates, samples)
                    candidates, samples = [], []
        if samples:
            yield self._train_batch(candidates, samples)

    def _train_batch(self, candidates: list, samples: list) -> dict:
        return {
            "candidate_rows": np.stack(candidates),
            "clicked_rows": self.histories[samples],
            "users": self.users[samples],
        }
