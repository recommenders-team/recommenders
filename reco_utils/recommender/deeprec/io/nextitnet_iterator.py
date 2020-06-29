# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import json
import pickle as pkl
import random
import os
import time

from reco_utils.recommender.deeprec.io.sequential_iterator import SequentialIterator
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

__all__ = ["NextItNetIterator"]


class NextItNetIterator(SequentialIterator):
    """Data loader for the NextItNet model.
    NextItNet requires a special type of data format. In training stage, each instance will produce (sequence_length * train_num_ngs) target items and labels, to let NextItNet output predictions of every item in a sequence except only of the last item.
    """

    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        * different from sequential iterator
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        self.col_spliter = col_spliter
        user_vocab, item_vocab, cate_vocab = (
            hparams.user_vocab,
            hparams.item_vocab,
            hparams.cate_vocab,
        )
        self.userdict, self.itemdict, self.catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )

        self.max_seq_length = hparams.max_seq_length
        self.batch_size = hparams.batch_size
        self.iter_data = dict()

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, None], name="label")
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.items = tf.placeholder(tf.int32, [None, None], name="items")
            self.cates = tf.placeholder(tf.int32, [None, None], name="cates")
            self.item_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.time = tf.placeholder(tf.float32, [None], name="time")
            self.time_diff = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_diff"
            )
            self.time_from_first_action = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            )
            self.time_to_now = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now"
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
        """Convert data into numpy arrays that are good for further model operation.
        * different from sequential_iterator
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return

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
                        i * (batch_num_ngs + 1) + index, -this_length:
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[
                        i * (batch_num_ngs + 1) + index, -this_length:
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    mask[i * (batch_num_ngs + 1) + index, -this_length:] = 1.0
                    time_diff_batch[
                        i * (batch_num_ngs + 1) + index, -this_length:
                    ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    time_from_first_action_batch[
                        i * (batch_num_ngs + 1) + index, -this_length:
                    ] = np.asarray(
                        time_from_first_action_list[i][-this_length:], dtype=np.float32
                    )
                    time_to_now_batch[
                        i * (batch_num_ngs + 1) + index, -this_length:
                    ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)

            for i in range(instance_cnt):
                positive_item = [
                    *item_history_batch_all[i * (batch_num_ngs + 1)][1:],
                    item_list[i],
                ]
                positive_item_cate = [
                    *item_cate_history_batch_all[i * (batch_num_ngs + 1)][1:],
                    item_cate_list[i],
                ]
                label_list_all.append([1] * max_seq_length_batch)
                # label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(positive_item_cate)

                count = 0
                while count < batch_num_ngs:
                    negative_item_list = []
                    negative_item_cate_list = []
                    count_inner = 1
                    while count_inner <= max_seq_length_batch:
                        random_value = random.randint(0, instance_cnt - 1)
                        negative_item = item_list[random_value]
                        if negative_item == positive_item[count_inner - 1]:
                            continue
                        negative_item_list.append(negative_item)
                        negative_item_cate_list.append(item_cate_list[random_value])
                        count_inner += 1

                    label_list_all.append([0] * max_seq_length_batch)
                    # label_list_all.append(0)
                    item_list_all.append(negative_item_list)
                    item_cate_list_all.append(negative_item_cate_list)
                    count += 1

            res = {}
            res["labels"] = np.asarray(
                label_list_all, dtype=np.float32
            )  # .reshape(-1,1)
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

            # print("label_list_all.shape: ", res["labels"].shape)
            # print("item_list_all.shape: ", res["items"].shape)

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
                item_history_batch_all[i, -this_length:] = item_history_batch[i][
                    -this_length:
                ]
                item_cate_history_batch_all[i, -this_length:] = item_cate_history_batch[
                    i
                ][-this_length:]
                mask[i, -this_length:] = 1.0
                time_diff_batch[i, -this_length:] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[
                    i, -this_length:
                ] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[i, -this_length:] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape([-1, 1])
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32).reshape([-1, 1])
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32).reshape([-1, 1])
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = np.asarray(time_list, dtype=np.float32)
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res
