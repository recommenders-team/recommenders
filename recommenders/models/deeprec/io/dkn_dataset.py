# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Streaming loaders for DKN and its item-to-item variant.

A news article is represented by the word indices and the entity indices of its
title, aligned position by position and all of the same length ``doc_size``. The
``news_feature_file`` holds one ``<news_id> <w1,w2,...> <e1,e2,...>`` line per
article.

Data files are read one mini-batch at a time, so files larger than memory can be
used as input.
"""

from __future__ import annotations

import numpy as np


def _parse_news_line(line: str) -> tuple[str, list[int], list[int]]:
    """Split a ``<news_id> <w1,w2,...> <e1,e2,...>`` line."""
    news_id, word_index, entity_index = line.strip().split(" ")
    return (
        news_id,
        [int(item) for item in word_index.split(",")],
        [int(item) for item in entity_index.split(",")],
    )


class NewsDataset:
    """Title word and entity indices of every news article, looked up by news ID."""

    def __init__(self, news_feature_file: str) -> None:
        """Load the news features.

        Args:
            news_feature_file (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line
                per article.
        """
        news_ids, words, entities = [], [], []
        with open(news_feature_file, "r") as rd:
            for line in rd:
                news_id, word_index, entity_index = _parse_news_line(line)
                news_ids.append(news_id)
                words.append(word_index)
                entities.append(entity_index)

        self.news_rows = {news_id: row for row, news_id in enumerate(news_ids)}
        # One all-zero row after the last article pads the short user histories.
        self.words = np.asarray(words + [[0] * len(words[0])], dtype=np.int64)
        self.entities = np.asarray(entities + [[0] * len(entities[0])], dtype=np.int64)
        if self.words.shape != self.entities.shape:
            raise ValueError(
                "Every article needs one entity index per word; got words of shape "
                "{0} and entities of shape {1}.".format(
                    self.words.shape, self.entities.shape
                )
            )

    def load_infer_data_from_file(self, infile: str, batch_size: int):
        """Read articles to embed, one mini-batch at a time.

        Args:
            infile (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line per
                article, the format of the news feature file.
            batch_size (int): Articles per mini-batch. The last batch of the file
                may hold fewer.

        Yields:
            dict, list:
            - ``words`` and ``entities``, both ``[batch_size, doc_size]``.
            - The news IDs of the batch.
        """
        news_ids, words, entities = [], [], []
        with open(infile, "r") as rd:
            for line in rd:
                news_id, word_index, entity_index = _parse_news_line(line)
                news_ids.append(news_id)
                words.append(word_index)
                entities.append(entity_index)

                if len(news_ids) == batch_size:
                    yield self._convert_infer_data(words, entities), news_ids
                    news_ids, words, entities = [], [], []
            if news_ids:
                yield self._convert_infer_data(words, entities), news_ids

    @staticmethod
    def _convert_infer_data(words: list, entities: list) -> dict:
        return {
            "words": np.asarray(words, dtype=np.int64),
            "entities": np.asarray(entities, dtype=np.int64),
        }


class DKNDataset(NewsDataset):
    """Mini-batch loader for DKN.

    Each line of a data file is ``<label> <user_id> <news_id>`` with an optional
    ``%<impression_id>`` suffix. The user's clicked articles come from the user
    history file.
    """

    def __init__(
        self, news_feature_file: str, user_history_file: str, history_size: int
    ) -> None:
        """Load the news features and the user histories.

        Args:
            news_feature_file (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line
                per article.
            user_history_file (str): One ``<user_id> <n1,n2,...>`` line per user,
                clicks in chronological order. A user with no clicks has a line
                holding only the ID.
            history_size (int): Clicked articles kept per user: the latest ones,
                with the zero padding row filling up shorter histories.
        """
        super().__init__(news_feature_file)
        padding_row = len(self.news_rows)
        self.user_history = {}
        with open(user_history_file, "r") as rd:
            for line in rd:
                user_id, _, history = line.strip().partition(" ")
                clicks = history.split(",") if history else []
                rows = [self.news_rows[news_id] for news_id in clicks[-history_size:]]
                rows += [padding_row] * (history_size - len(rows))
                self.user_history[user_id] = np.asarray(rows, dtype=np.int64)

    def parser_one_line(self, line: str) -> tuple[float, str, str, str | int]:
        """Parse one string line.

        Args:
            line (str): A string indicating one instance.

        Returns:
            float, str, str, str | int:
            - The label.
            - The user ID.
            - The candidate news ID.
            - The impression ID, or `0` when the line carries none.
        """
        impression_id = 0
        words = line.strip().split("%")
        if len(words) == 2:
            impression_id = words[1].strip()

        label, user_id, news_id = words[0].strip().split(" ")
        return float(label), user_id, news_id, impression_id

    def load_data_from_file(self, infile: str, batch_size: int):
        """Read and parse data from a file, one mini-batch at a time.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_size (int): Instances per mini-batch. The last batch of the file
                may hold fewer.

        Yields:
            dict, list:
            - ``labels`` ``[batch_size, 1]``; ``candidate_words`` and
              ``candidate_entities`` ``[batch_size, doc_size]``; ``clicked_words``
              and ``clicked_entities`` ``[batch_size, history_size, doc_size]``.
            - The impression IDs of the batch.
        """
        labels, candidates, histories, impression_ids = [], [], [], []
        with open(infile, "r") as rd:
            for line in rd:
                label, user_id, news_id, impression_id = self.parser_one_line(line)
                labels.append(label)
                candidates.append(self.news_rows[news_id])
                histories.append(self.user_history[user_id])
                impression_ids.append(impression_id)

                if len(labels) == batch_size:
                    yield self._convert_data(
                        labels, candidates, histories
                    ), impression_ids
                    labels, candidates, histories, impression_ids = [], [], [], []
            if labels:
                yield self._convert_data(labels, candidates, histories), impression_ids

    def _convert_data(self, labels: list, candidates: list, histories: list) -> dict:
        histories = np.stack(histories)
        return {
            "labels": np.asarray(labels, dtype=np.float32).reshape(-1, 1),
            "candidate_words": self.words[candidates],
            "candidate_entities": self.entities[candidates],
            "clicked_words": self.words[histories],
            "clicked_entities": self.entities[histories],
        }


class DKNItem2ItemDataset(NewsDataset):
    """Mini-batch loader for DKN's item-to-item variant.

    A data file holds one news ID per line, in groups of ``neg_num + 2``
    consecutive lines: the source article, the related target article, then
    ``neg_num`` unrelated ones.
    """

    def __init__(self, news_feature_file: str, neg_num: int) -> None:
        """Load the news features.

        Args:
            news_feature_file (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line
                per article.
            neg_num (int): Unrelated articles per group.
        """
        super().__init__(news_feature_file)
        self.group_size = neg_num + 2

    def load_data_from_file(self, infile: str, batch_size: int):
        """Read and parse data from a file, one mini-batch of groups at a time.

        Args:
            infile (str): Text input file, one news ID per line.
            batch_size (int): Groups per mini-batch. The last batch of the file may
                hold fewer.

        Yields:
            dict, list:
            - ``words`` and ``entities``, both
              ``[batch_size, neg_num + 2, doc_size]``.
            - The news IDs of the batch, in file order.
        """
        news_ids = []
        with open(infile, "r") as rd:
            for line in rd:
                news_ids.append(line.strip())
                if len(news_ids) == batch_size * self.group_size:
                    yield self._convert_data(news_ids), news_ids
                    news_ids = []
            if news_ids:
                if len(news_ids) % self.group_size:
                    raise ValueError(
                        "{0} ends with an incomplete group: {1} trailing lines for "
                        "groups of neg_num + 2 = {2}.".format(
                            infile,
                            len(news_ids) % self.group_size,
                            self.group_size,
                        )
                    )
                yield self._convert_data(news_ids), news_ids

    def _convert_data(self, news_ids: list) -> dict:
        rows = np.asarray(
            [self.news_rows[news_id] for news_id in news_ids], dtype=np.int64
        ).reshape(-1, self.group_size)
        return {"words": self.words[rows], "entities": self.entities[rows]}
