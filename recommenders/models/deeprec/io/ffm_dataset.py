# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Streaming loader for FFM-format data, the input format of xDeepFM.

Port of the TF-1.x ``FFMTextIterator``: the text parsing is unchanged, but batches
come out as numpy arrays laid out for ``torch.nn.functional.embedding_bag`` instead
of as a graph ``feed_dict``. Like the TF version it reads one mini-batch at a time,
so files larger than memory can be used as input.

Each line is ``<label> <field>:<feature>:<value> ...`` with an optional
``%<impression_id>`` suffix. Field and feature indices are 1-based in the file and
are converted to 0-based here.

A batch flattens the feature triples of all its instances and sorts them by
``(instance, field)``:

* ``feat_ids`` / ``feat_values`` -- the ``(feature, value)`` pairs, ``[nnz]``.
* ``dnn_offsets`` -- start of each of the ``batch_size * field_count`` per-field
  bags, ``[batch_size * field_count]``. Sum-pooling over those bags reproduces TF's
  ``embedding_lookup_sparse(combiner="sum")``; every ``field_count``-th offset gives
  the per-instance bags that the linear and FM parts sum over.
"""

from __future__ import annotations

import numpy as np


class FFMDataset:
    """Mini-batch loader for FFM-format text files."""

    def __init__(
        self, field_count: int, col_spliter: str = " ", ID_spliter: str = "%"
    ) -> None:
        """Initialize the loader.

        Args:
            field_count (int): Number of fields per instance. Every instance is
                bagged into exactly this many per-field groups.
            col_spliter (str): Column splitter in one line.
            ID_spliter (str): Impression-ID splitter in one line.
        """
        self.field_count = field_count
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter

    def parser_one_line(self, line: str) -> tuple[float, list, str]:
        """Parse one string line into feature values.

        Args:
            line (str): A string indicating one instance.

        Returns:
            float, list, str:
            - The label.
            - The `[field_idx, feature_idx, feature_value]` triples, 0-based.
            - The impression ID, or `0` when the line carries none.
        """
        impression_id = 0
        words = line.strip().split(self.ID_spliter)
        if len(words) == 2:
            impression_id = words[1].strip()

        cols = words[0].strip().split(self.col_spliter)

        label = float(cols[0])

        features = []
        for word in cols[1:]:
            if not word.strip():
                continue
            tokens = word.split(":")
            features.append([int(tokens[0]) - 1, int(tokens[1]) - 1, float(tokens[2])])

        return label, features, impression_id

    def load_data_from_file(self, infile: str, batch_size: int):
        """Read and parse data from a file, one mini-batch at a time.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_size (int): Number of instances per mini-batch.

        Yields:
            dict, list, int:
            - The batch arrays, see the module docstring.
            - The impression IDs of the batch.
            - The number of instances in the batch, which is smaller than
              `batch_size` for the last batch of the file.
        """
        label_list = []
        features_list = []
        impression_id_list = []

        with open(infile, "r") as rd:
            for line in rd:
                label, features, impression_id = self.parser_one_line(line)

                label_list.append(label)
                features_list.append(features)
                impression_id_list.append(impression_id)

                if len(label_list) == batch_size:
                    yield (
                        self._convert_data(label_list, features_list),
                        impression_id_list,
                        batch_size,
                    )
                    label_list = []
                    features_list = []
                    impression_id_list = []
            if label_list:
                yield (
                    self._convert_data(label_list, features_list),
                    impression_id_list,
                    len(label_list),
                )

    def _convert_data(self, labels: list, features: list) -> dict:
        """Flatten a batch of parsed lines into the arrays the model consumes.

        Args:
            labels (list): The ground-truth labels of the batch.
            features (list): Per instance, the list of
                `[field_idx, feature_idx, feature_value]` triples.

        Returns:
            dict: `labels`, `feat_ids`, `feat_values` and `dnn_offsets`.
        """
        instance_count = len(labels)
        bag_count = instance_count * self.field_count

        rows = []
        feat_ids = []
        feat_values = []
        for i, instance in enumerate(features):
            for field_idx, feature_idx, feature_value in instance:
                rows.append(i * self.field_count + field_idx)
                feat_ids.append(feature_idx)
                feat_values.append(feature_value)

        rows = np.asarray(rows, dtype=np.int64)
        counts = np.bincount(rows, minlength=bag_count)
        if counts.size != bag_count:
            raise ValueError(
                "Found a field index >= field_count ({0}); check the data format "
                "and the FIELD_COUNT setting.".format(self.field_count)
            )
        order = np.argsort(rows, kind="stable")

        return {
            "labels": np.asarray(labels, dtype=np.float32).reshape(-1, 1),
            "feat_ids": np.asarray(feat_ids, dtype=np.int64)[order],
            "feat_values": np.asarray(feat_values, dtype=np.float32)[order],
            "dnn_offsets": np.cumsum(counts) - counts,
        }
