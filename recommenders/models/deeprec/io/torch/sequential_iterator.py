# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from __future__ import annotations

import random
from typing import Iterator

import numpy as np

from recommenders.models.deeprec.deeprec_utils import load_dict


__all__ = ["SequentialIterator"]


class SequentialIterator:
    """TF-free sequential iterator for PyTorch sequential models.

    Parses Amazon-reviews-style tab-separated input files and yields batches as
    ``dict[str, np.ndarray]``. Unlike the TF
    :class:`recommenders.models.deeprec.io.sequential_iterator.SequentialIterator`,
    this iterator creates no TF graph and returns plain numpy arrays so
    callers can convert them to ``torch.Tensor`` directly.

    The parsing logic (label / ids / item-history / cate-history / time
    features) and the in-batch negative-sampling layout match the TF iterator
    exactly so models can be compared head-to-head.
    """

    def __init__(
        self,
        user_vocab: str,
        item_vocab: str,
        cate_vocab: str,
        max_seq_length: int,
        batch_size: int,
        col_spliter: str = "\t",
    ) -> None:
        """Build the iterator.

        Args:
            user_vocab (str): Path to the pickled user-to-id mapping.
            item_vocab (str): Path to the pickled item-to-id mapping.
            cate_vocab (str): Path to the pickled category-to-id mapping.
            max_seq_length (int): Maximum length of the history sequence;
                shorter sequences are right-padded with 0.
            batch_size (int): Number of positive instances per batch (before
                in-batch negative expansion).
            col_spliter (str): Column delimiter in the input file.
        """
        self.col_spliter = col_spliter
        self.userdict = load_dict(user_vocab)
        self.itemdict = load_dict(item_vocab)
        self.catedict = load_dict(cate_vocab)

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.iter_data: dict[str, list] = {}

    @property
    def user_vocab_length(self) -> int:
        return len(self.userdict)

    @property
    def item_vocab_length(self) -> int:
        return len(self.itemdict)

    @property
    def cate_vocab_length(self) -> int:
        return len(self.catedict)

    def parse_file(self, input_file: str) -> list:
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line: str) -> tuple:
        """Parse one line into the same 10-tuple SequentialIterator produces."""
        words = line.strip().split(self.col_spliter)
        label = int(words[0])
        user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
        item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
        item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
        current_time = float(words[4])

        item_history_sequence = [
            self.itemdict[item] if item in self.itemdict else 0
            for item in words[5].strip().split(",")
        ]
        cate_history_sequence = [
            self.catedict[cate] if cate in self.catedict else 0
            for cate in words[6].strip().split(",")
        ]
        time_history_sequence = [float(i) for i in words[7].strip().split(",")]

        time_range = 3600 * 24

        # The three time features below (time_diff, time_from_first_action,
        # time_to_now) are emitted for the time-aware sequential models
        # (SLi-Rec consumes them via Time4LSTMCell). GRU intentionally ignores
        # them; its forward() and _to_device() do not reference these keys,
        # so the per-batch cost is Python-side computation only (no GPU
        # transfer). Kept here so the same iterator can serve future
        # SLi-Rec / time-aware migrations without forking the parser.
        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (
                time_history_sequence[i + 1] - time_history_sequence[i]
            ) / time_range
            time_diff.append(max(diff, 0.5))
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        time_diff.append(max(last_diff, 0.5))
        time_diff = np.log(time_diff)

        first_time = time_history_sequence[0]
        time_from_first_action = [
            max((t - first_time) / time_range, 0.5)
            for t in time_history_sequence[1:]
        ]
        time_from_first_action.append(max((current_time - first_time) / time_range, 0.5))
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = [
            max((current_time - t) / time_range, 0.5) for t in time_history_sequence
        ]
        time_to_now = np.log(time_to_now)

        return (
            label,
            user_id,
            item_id,
            item_cate,
            item_history_sequence,
            cate_history_sequence,
            current_time,
            time_diff,
            time_from_first_action,
            time_to_now,
        )

    def load_data_from_file(
        self,
        infile: str,
        batch_num_ngs: int = 0,
        min_seq_length: int = 1,
    ) -> Iterator[dict[str, np.ndarray] | None]:
        """Read and parse data from a file, yielding batches of numpy arrays.

        Args:
            infile (str): Path to a train / valid / test file.
            batch_num_ngs (int): Number of in-batch negative samples per
                positive instance. 0 disables negative sampling (eval mode).
            min_seq_length (int): Sequences with history shorter than this are
                skipped.

        Yields:
            dict[str, np.ndarray] | None: One batch per yield. ``None`` is
            yielded when a tail batch is too small for negative sampling, to
            match :class:`SequentialIterator` semantics.
        """
        if infile not in self.iter_data:
            self.iter_data[infile] = self.parse_file(infile)
        lines = self.iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []

        cnt = 0
        for line in lines:
            if not line:
                continue
            (
                label,
                user_id,
                item_id,
                item_cate,
                item_history_sequence,
                item_cate_history_sequence,
                current_time,
                time_diff,
                time_from_first_action,
                time_to_now,
            ) = line
            if len(item_history_sequence) < min_seq_length:
                continue

            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)

            cnt += 1
            if cnt == self.batch_size:
                yield self._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    batch_num_ngs,
                )
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cnt = 0

        if cnt > 0:
            yield self._convert_data(
                label_list,
                user_list,
                item_list,
                item_cate_list,
                item_history_batch,
                item_cate_history_batch,
                time_list,
                time_diff_list,
                time_from_first_action_list,
                time_to_now_list,
                batch_num_ngs,
            )

    def _convert_data(
        self,
        label_list,
        user_list,
        item_list,
        item_cate_list,
        item_history_batch,
        item_cate_history_batch,
        time_list,
        time_diff_list,
        time_from_first_action_list,
        time_to_now_list,
        batch_num_ngs: int,
    ) -> dict[str, np.ndarray] | None:
        """Build a batch dict, expanding with in-batch negatives when requested.

        Layout for ``batch_num_ngs > 0`` matches SequentialIterator: each
        positive at row ``i * (1 + batch_num_ngs)`` is followed by its
        ``batch_num_ngs`` negatives. Loss code relies on this layout to
        reshape ``logit`` into ``(-1, 1 + batch_num_ngs)``.
        """
        max_seq = self.max_seq_length

        if batch_num_ngs:
            instance_cnt = len(label_list)
            # SequentialIterator skips tiny tail batches to avoid degenerate
            # negative sampling; we preserve that semantic by returning None.
            if instance_cnt < 5:
                return None

            group = batch_num_ngs + 1
            n_rows = instance_cnt * group

            user_all = np.repeat(np.asarray(user_list, dtype=np.int32), group)
            time_all = np.repeat(np.asarray(time_list, dtype=np.float32), group)

            item_history_all = np.zeros((n_rows, max_seq), dtype=np.int32)
            item_cate_history_all = np.zeros((n_rows, max_seq), dtype=np.int32)
            time_diff_batch = np.zeros((n_rows, max_seq), dtype=np.float32)
            time_from_first_action_batch = np.zeros((n_rows, max_seq), dtype=np.float32)
            time_to_now_batch = np.zeros((n_rows, max_seq), dtype=np.float32)
            mask = np.zeros((n_rows, max_seq), dtype=np.float32)

            for i in range(instance_cnt):
                this_length = min(len(item_history_batch[i]), max_seq)
                rows = slice(i * group, (i + 1) * group)
                item_history_all[rows, :this_length] = item_history_batch[i][-this_length:]
                item_cate_history_all[rows, :this_length] = item_cate_history_batch[i][-this_length:]
                mask[rows, :this_length] = 1.0
                time_diff_batch[rows, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[rows, :this_length] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[rows, :this_length] = time_to_now_list[i][-this_length:]

            label_all = []
            item_all = []
            item_cate_all = []
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_all.append(1)
                item_all.append(positive_item)
                item_cate_all.append(item_cate_list[i])
                count = 0
                attempts = 0
                while count < batch_num_ngs:
                    attempts += 1
                    if attempts > 100 * batch_num_ngs:
                        raise ValueError(
                            "could not sample enough distinct negatives; "
                            "batch has too few unique items"
                        )
                    j = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[j]
                    if negative_item == positive_item:
                        continue
                    label_all.append(0)
                    item_all.append(negative_item)
                    item_cate_all.append(item_cate_list[j])
                    count += 1

            return {
                "labels": np.asarray(label_all, dtype=np.float32).reshape(-1, 1),
                "users": user_all,
                "items": np.asarray(item_all, dtype=np.int32),
                "cates": np.asarray(item_cate_all, dtype=np.int32),
                "item_history": item_history_all,
                "item_cate_history": item_cate_history_all,
                "mask": mask,
                "time": time_all,
                "time_diff": time_diff_batch,
                "time_from_first_action": time_from_first_action_batch,
                "time_to_now": time_to_now_batch,
            }

        instance_cnt = len(label_list)
        item_history_all = np.zeros((instance_cnt, max_seq), dtype=np.int32)
        item_cate_history_all = np.zeros((instance_cnt, max_seq), dtype=np.int32)
        time_diff_batch = np.zeros((instance_cnt, max_seq), dtype=np.float32)
        time_from_first_action_batch = np.zeros((instance_cnt, max_seq), dtype=np.float32)
        time_to_now_batch = np.zeros((instance_cnt, max_seq), dtype=np.float32)
        mask = np.zeros((instance_cnt, max_seq), dtype=np.float32)

        for i in range(instance_cnt):
            this_length = min(len(item_history_batch[i]), max_seq)
            item_history_all[i, :this_length] = item_history_batch[i][-this_length:]
            item_cate_history_all[i, :this_length] = item_cate_history_batch[i][-this_length:]
            mask[i, :this_length] = 1.0
            time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
            time_from_first_action_batch[i, :this_length] = time_from_first_action_list[i][-this_length:]
            time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

        return {
            "labels": np.asarray(label_list, dtype=np.float32).reshape(-1, 1),
            "users": np.asarray(user_list, dtype=np.int32),
            "items": np.asarray(item_list, dtype=np.int32),
            "cates": np.asarray(item_cate_list, dtype=np.int32),
            "item_history": item_history_all,
            "item_cate_history": item_cate_history_all,
            "mask": mask,
            "time": np.asarray(time_list, dtype=np.float32),
            "time_diff": time_diff_batch,
            "time_from_first_action": time_from_first_action_batch,
            "time_to_now": time_to_now_batch,
        }
