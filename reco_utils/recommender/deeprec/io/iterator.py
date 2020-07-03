# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import abc


class BaseIterator(object):
    @abc.abstractmethod
    def parser_one_line(self, line):
        pass

    @abc.abstractmethod
    def load_data_from_file(self, infile):
        pass

    @abc.abstractmethod
    def _convert_data(self, labels, features):
        pass

    @abc.abstractmethod
    def gen_feed_dict(self, data_dict):
        pass


class FFMTextIterator(BaseIterator):
    """Data loader for FFM format based models, such as xDeepFM.
    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.
    """

    def __init__(self, hparams, graph, col_spliter=" ", ID_spliter="%"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column splitter in one line.
            ID_spliter (str): ID splitter in one line.
        """
        self.feature_cnt = hparams.FEATURE_COUNT
        self.field_cnt = hparams.FIELD_COUNT
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.fm_feat_indices = tf.placeholder(
                tf.int64, [None, 2], name="fm_feat_indices"
            )
            self.fm_feat_values = tf.placeholder(
                tf.float32, [None], name="fm_feat_values"
            )
            self.fm_feat_shape = tf.placeholder(tf.int64, [None], name="fm_feat_shape")
            self.dnn_feat_indices = tf.placeholder(
                tf.int64, [None, 2], name="dnn_feat_indices"
            )
            self.dnn_feat_values = tf.placeholder(
                tf.int64, [None], name="dnn_feat_values"
            )
            self.dnn_feat_weights = tf.placeholder(
                tf.float32, [None], name="dnn_feat_weights"
            )
            self.dnn_feat_shape = tf.placeholder(
                tf.int64, [None], name="dnn_feat_shape"
            )

    def parser_one_line(self, line):
        """Parse one string line into feature values.
        
        Args:
            line (str): a string indicating one instance

        Returns:
            list: Parsed results,including label, features and impression_id

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

    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
        label_list = []
        features_list = []
        impression_id_list = []
        cnt = 0

        with tf.gfile.GFile(infile, "r") as rd:
            while True:
                line = rd.readline()
                if not line:
                    break

                label, features, impression_id = self.parser_one_line(line)

                features_list.append(features)
                label_list.append(label)
                impression_id_list.append(impression_id)

                cnt += 1
                if cnt == self.batch_size:
                    res = self._convert_data(label_list, features_list)
                    yield self.gen_feed_dict(res), impression_id_list, self.batch_size
                    label_list = []
                    features_list = []
                    impression_id_list = []
                    cnt = 0
            if cnt > 0:
                res = self._convert_data(label_list, features_list)
                yield self.gen_feed_dict(res), impression_id_list, cnt

    def _convert_data(self, labels, features):
        """Convert data into numpy arrays that are good for further operation.
        
        Args:
            labels (list): a list of ground-truth labels.
            features (list): a 3-dimensional list, carrying a list (batch_size) of feature array,
                    where each feature array is a list of [field_idx, feature_idx, feature_value] tuple.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        dim = self.feature_cnt
        FIELD_COUNT = self.field_cnt
        instance_cnt = len(labels)

        fm_feat_indices = []
        fm_feat_values = []
        fm_feat_shape = [instance_cnt, dim]

        dnn_feat_indices = []
        dnn_feat_values = []
        dnn_feat_weights = []
        dnn_feat_shape = [instance_cnt * FIELD_COUNT, -1]

        for i in range(instance_cnt):
            m = len(features[i])
            dnn_feat_dic = {}
            for j in range(m):
                fm_feat_indices.append([i, features[i][j][1]])
                fm_feat_values.append(features[i][j][2])
                if features[i][j][0] not in dnn_feat_dic:
                    dnn_feat_dic[features[i][j][0]] = 0
                else:
                    dnn_feat_dic[features[i][j][0]] += 1
                dnn_feat_indices.append(
                    [
                        i * FIELD_COUNT + features[i][j][0],
                        dnn_feat_dic[features[i][j][0]],
                    ]
                )
                dnn_feat_values.append(features[i][j][1])
                dnn_feat_weights.append(features[i][j][2])
                if dnn_feat_shape[1] < dnn_feat_dic[features[i][j][0]]:
                    dnn_feat_shape[1] = dnn_feat_dic[features[i][j][0]]
        dnn_feat_shape[1] += 1

        sorted_index = sorted(
            range(len(dnn_feat_indices)),
            key=lambda k: (dnn_feat_indices[k][0], dnn_feat_indices[k][1]),
        )

        res = {}
        res["fm_feat_indices"] = np.asarray(fm_feat_indices, dtype=np.int64)
        res["fm_feat_values"] = np.asarray(fm_feat_values, dtype=np.float32)
        res["fm_feat_shape"] = np.asarray(fm_feat_shape, dtype=np.int64)
        res["labels"] = np.asarray([[label] for label in labels], dtype=np.float32)

        res["dnn_feat_indices"] = np.asarray(dnn_feat_indices, dtype=np.int64)[
            sorted_index
        ]
        res["dnn_feat_values"] = np.asarray(dnn_feat_values, dtype=np.int64)[
            sorted_index
        ]
        res["dnn_feat_weights"] = np.asarray(dnn_feat_weights, dtype=np.float32)[
            sorted_index
        ]
        res["dnn_feat_shape"] = np.asarray(dnn_feat_shape, dtype=np.int64)
        return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        feed_dict = {
            self.labels: data_dict["labels"],
            self.fm_feat_indices: data_dict["fm_feat_indices"],
            self.fm_feat_values: data_dict["fm_feat_values"],
            self.fm_feat_shape: data_dict["fm_feat_shape"],
            self.dnn_feat_indices: data_dict["dnn_feat_indices"],
            self.dnn_feat_values: data_dict["dnn_feat_values"],
            self.dnn_feat_weights: data_dict["dnn_feat_weights"],
            self.dnn_feat_shape: data_dict["dnn_feat_shape"],
        }
        return feed_dict
