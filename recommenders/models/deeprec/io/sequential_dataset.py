# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""PyTorch data loader for SLi-Rec (and other sequential recommenders).

This is a faithful port of
``recommenders.models.deeprec.io.sequential_iterator.SequentialIterator`` with the
TF-1.x placeholders and ``gen_feed_dict`` removed — parsing and batch assembly are
already pure Python/NumPy, so the arrays produced here are byte-identical to the TF
iterator's. The PyTorch model converts each yielded numpy dict to tensors.

Kept exactly as in the TF iterator: the 8-column tab-separated line format, OOV/PAD
index 0, the three log-scaled time features (with a 0.5 floor applied *before* the
log), left-align + most-recent truncation to ``max_seq_length``, in-batch negative
sampling with a positive-first ``[1, 0, ...]`` ordering, the ``< 5`` instances -> skip
rule during training, and a per-epoch shuffle only when sampling.
"""

from __future__ import annotations

import random

import numpy as np

from recommenders.models.deeprec.deeprec_utils import load_dict

__all__ = ["SequentialDataset"]


class SequentialDataset:
    """Parse sequential-recommendation files into padded NumPy batches.

    Mirrors ``SequentialIterator``'s public data method ``load_data_from_file``; the
    PyTorch SLi-Rec model builds one of these internally from the vocabulary paths.
    """

    def __init__(
        self, user_vocab, item_vocab, cate_vocab, max_seq_length, col_spliter="\t"
    ):
        """Initialize the loader.

        Args:
            user_vocab (str): Path to the user token→id vocabulary pickle.
            item_vocab (str): Path to the item token→id vocabulary pickle.
            cate_vocab (str): Path to the category token→id vocabulary pickle.
            max_seq_length (int): Maximum history length (padding/truncation target).
            col_spliter (str): Column splitter within a line.
        """
        self.col_spliter = col_spliter
        self.userdict, self.itemdict, self.catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )
        self.max_seq_length = max_seq_length
        self.iter_data = dict()

    def parse_file(self, input_file):
        """Parse every line of ``input_file`` into feature tuples."""
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        """Parse one instance line into feature values.

        Returns:
            tuple: ``(label, user_id, item_id, item_cate, item_history_sequence,
            cate_history_sequence, current_time, time_diff, time_from_first_action,
            time_to_now)``.
        """
        words = line.strip().split(self.col_spliter)
        label = int(words[0])
        user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
        item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
        item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
        current_time = float(words[4])

        item_history_sequence = []
        cate_history_sequence = []

        item_history_words = words[5].strip().split(",")
        for item in item_history_words:
            item_history_sequence.append(
                self.itemdict[item] if item in self.itemdict else 0
            )

        cate_history_words = words[6].strip().split(",")
        for cate in cate_history_words:
            cate_history_sequence.append(
                self.catedict[cate] if cate in self.catedict else 0
            )

        time_history_words = words[7].strip().split(",")
        time_history_sequence = [float(i) for i in time_history_words]

        time_range = 3600 * 24

        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (
                time_history_sequence[i + 1] - time_history_sequence[i]
            ) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)

        first_time = time_history_sequence[0]
        time_from_first_action = [
            (t - first_time) / time_range for t in time_history_sequence[1:]
        ]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
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
        self, infile, batch_size, batch_num_ngs=0, min_seq_length=1
    ):
        """Read and parse ``infile``, yielding one NumPy-dict batch at a time.

        Args:
            infile (str): Input file; each line is one instance.
            batch_size (int): Number of positive instances accumulated per batch.
            batch_num_ngs (int): In-batch negatives per positive. ``0`` disables
                sampling (evaluation path) and preserves file order.
            min_seq_length (int): Instances with a shorter history are skipped.

        Yields:
            dict | None: A batch dict of NumPy arrays, or ``None`` for a skipped batch.
        """
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

        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

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
            if cnt == batch_size:
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
        batch_num_ngs,
    ):
        """Assemble padded NumPy arrays for one batch.

        Returns ``None`` (skip) when ``batch_num_ngs`` is set and fewer than 5
        instances are available, matching the TF iterator.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return None

            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            time_list_all = np.asarray(
                [[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32
            ).flatten()

            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_from_first_action_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            mask = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            ).astype("float32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    time_diff_batch[i * (batch_num_ngs + 1) + index, :this_length] = (
                        np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    )
                    time_from_first_action_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        time_from_first_action_list[i][-this_length:], dtype=np.float32
                    )
                    time_to_now_batch[i * (batch_num_ngs + 1) + index, :this_length] = (
                        np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)
                    )

            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
            res["users"] = user_list_all
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = time_list_all
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res

        else:
            instance_cnt = len(label_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            time_from_first_action_batch = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][
                    -this_length:
                ]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[
                    i
                ][-this_length:]
                mask[i, :this_length] = 1.0
                time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[i, :this_length] = (
                    time_from_first_action_list[i][-this_length:]
                )
                time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = np.asarray(time_list, dtype=np.float32)
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res
