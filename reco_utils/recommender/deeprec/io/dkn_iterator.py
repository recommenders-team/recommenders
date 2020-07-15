# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

from reco_utils.recommender.deeprec.io.iterator import BaseIterator


__all__ = ["DKNTextIterator"]


class DKNTextIterator(BaseIterator):
    """Data loader for the DKN model.
    DKN requires a special type of data format, where each instance contains a label, the candidate news article,
    and user's clicked news article. Articles are represented by title words and title entities. Words and entities
    are aligned.

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.
    """

    def __init__(self, hparams, graph, col_spliter=" ", ID_spliter="%"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.doc_size = hparams.doc_size
        self.his_size = hparams.his_size

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.candidate_news_index_batch = tf.placeholder(
                tf.int64, [self.batch_size, self.doc_size], name="candidate_news_index"
            )
            self.click_news_index_batch = tf.placeholder(
                tf.int64, [self.batch_size, self.his_size, self.doc_size], name="click_news_index"
            )
            self.candidate_news_entity_index_batch = tf.placeholder(
                tf.int64,
                [self.batch_size, self.doc_size],
                name="candidate_news_entity_index",
            )
            self.click_news_entity_index_batch = tf.placeholder(
                tf.int64,
                [self.batch_size, self.his_size, self.doc_size],
                name="click_news_entity_index",
            )
        self.news_word_index = {}
        self.news_entity_index = {}
        with tf.gfile.GFile(hparams.news_feature_file, "r") as rd:
            for line in rd:
                newsid, word_index, entity_index = line.strip().split(col_spliter)
                self.news_word_index[newsid] = [int(item) for item in word_index.split(',')]
                self.news_entity_index[newsid] = [int(item) for item in entity_index.split(',')]
        self.user_history = {}
        with tf.gfile.GFile(hparams.user_history_file, "r") as rd:
            for line in rd:
                userid, user_history_string = line.strip().split(col_spliter)
                user_history = user_history_string.split(",")
                click_news_index = []
                click_news_entity_index = []
                if len(user_history) > self.his_size:
                    user_history = user_history[-self.his_size:]
                for newsid in user_history:
                    click_news_index.append(self.news_word_index[newsid])
                    click_news_entity_index.append(self.news_entity_index[newsid])
                for i in range(self.his_size- len(user_history)):
                    click_news_index.append(np.zeros(self.doc_size))
                    click_news_entity_index.append(np.zeros(self.doc_size))
                self.user_history[userid] = (click_news_index, click_news_entity_index)

    def parser_one_line(self, line):
        """Parse one string line into feature values.
        
        Args:
            line (str): a string indicating one instance

        Returns:
            list: Parsed results including label, candidate_news_index, candidate_news_val, click_news_index, click_news_val,
            candidate_news_entity_index, click_news_entity_index, impression_id

        """
        impression_id = 0
        words = line.strip().split(self.ID_spliter)
        if len(words) == 2:
            impression_id = words[1].strip()

        cols = words[0].strip().split(self.col_spliter)
        label = float(cols[0])

        userid = cols[1]
        candidate_news = cols[2]

        candidate_news_index = self.news_word_index[candidate_news]
        candidate_news_entity_index = self.news_entity_index[candidate_news]
        click_news_index = self.user_history[userid][0]
        click_news_entity_index = self.user_history[userid][1]

        return (
            label,
            candidate_news_index,
            click_news_index,
            candidate_news_entity_index,
            click_news_entity_index,
            impression_id,
        )

    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
            List: impression id list
            Int: size of the data in a batch
        """
        candidate_news_index_batch = []
        click_news_index_batch = []
        candidate_news_entity_index_batch = []
        click_news_entity_index_batch = []
        label_list = []
        impression_id_list = []
        cnt = 0

        with tf.gfile.GFile(infile, "r") as rd:
            for line in rd:
                (
                    label,
                    candidate_news_index,
                    click_news_index,
                    candidate_news_entity_index,
                    click_news_entity_index,
                    impression_id,
                ) = self.parser_one_line(line)

                candidate_news_index_batch.append(candidate_news_index)
                click_news_index_batch.append(click_news_index)
                candidate_news_entity_index_batch.append(candidate_news_entity_index)
                click_news_entity_index_batch.append(click_news_entity_index)
                label_list.append(label)
                impression_id_list.append(impression_id)

                cnt += 1
                if cnt >= self.batch_size:
                    res = self._convert_data(
                        label_list,
                        candidate_news_index_batch,
                        click_news_index_batch,
                        candidate_news_entity_index_batch,
                        click_news_entity_index_batch,
                        impression_id_list,
                    )
                    data_size = self.batch_size
                    yield self.gen_feed_dict(res), impression_id_list, data_size
                    candidate_news_index_batch = []
                    click_news_index_batch = []
                    candidate_news_entity_index_batch = []
                    click_news_entity_index_batch = []
                    label_list = []
                    impression_id_list = []
                    cnt = 0
            if cnt > 0:
                data_size = cnt
                while cnt < self.batch_size:
                    candidate_news_index_batch.append(
                        candidate_news_index_batch[cnt % data_size]
                    )
                    click_news_index_batch.append(
                        click_news_index_batch[cnt % data_size]
                    )
                    candidate_news_entity_index_batch.append(
                        candidate_news_entity_index_batch[cnt % data_size]
                    )
                    click_news_entity_index_batch.append(
                        click_news_entity_index_batch[cnt % data_size]
                    )
                    label_list.append(label_list[cnt % data_size])
                    impression_id_list.append(impression_id_list[cnt % data_size])
                    cnt += 1
                res = self._convert_data(
                    label_list,
                    candidate_news_index_batch,
                    click_news_index_batch,
                    candidate_news_entity_index_batch,
                    click_news_entity_index_batch,
                    impression_id_list,
                )
                yield self.gen_feed_dict(res), impression_id_list, data_size

    def load_infer_data_from_file(self, infile):
        """Read and parse data from a file for infer document embedding.

        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
            List: news id list
            Int: size of the data in a batch
        """
        newsid_list = []
        candidate_news_index_batch = []
        candidate_news_entity_index_batch = []
        cnt = 0
        with tf.gfile.GFile(infile, "r") as rd:
            for line in rd:
                newsid, word_index, entity_index = line.strip().split(" ")
                newsid_list.append(newsid)
                candidate_news_index = []
                candidate_news_entity_index = []
                for item in word_index.split(","):
                    candidate_news_index.append(int(item))
                for item in entity_index.split(","):
                    candidate_news_entity_index.append(int(item))

                candidate_news_index_batch.append(candidate_news_index)
                candidate_news_entity_index_batch.append(candidate_news_entity_index)

                cnt += 1
                if cnt >= self.batch_size:
                    res = self._convert_infer_data(
                        candidate_news_index_batch,
                        candidate_news_entity_index_batch,
                    )
                    data_size = self.batch_size
                    yield self.gen_infer_feed_dict(res), newsid_list, data_size
                    candidate_news_index_batch = []
                    candidate_news_entity_index_batch = []
                    newsid_list = []
                    cnt = 0

            if cnt > 0:
                data_size = cnt
                while cnt < self.batch_size:
                    candidate_news_index_batch.append(
                        candidate_news_index_batch[cnt % data_size]
                    )
                    candidate_news_entity_index_batch.append(
                        candidate_news_entity_index_batch[cnt % data_size]
                    )
                    cnt += 1
                res = self._convert_infer_data(
                    candidate_news_index_batch,
                    candidate_news_entity_index_batch,
                )
                yield self.gen_infer_feed_dict(res), newsid_list, data_size

    def _convert_data(
        self,
        label_list,
        candidate_news_index_batch,
        click_news_index_batch,
        candidate_news_entity_index_batch,
        click_news_entity_index_batch,
        impression_id_list,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            candidate_news_index_batch (list): the candidate news article's words indices
            click_news_index_batch (list): words indices for user's clicked news articles
            candidate_news_entity_index_batch (list): the candidate news article's entities indices
            click_news_entity_index_batch (list): the user's clicked news article's entities indices
            impression_id_list (list) : the session's impression indices

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        res = {}
        res["labels"] = np.asarray([[label] for label in label_list], dtype=np.float32)
        res["candidate_news_index_batch"] = np.asarray(
            candidate_news_index_batch, dtype=np.int64
        )
        res["click_news_index_batch"] = np.asarray(
            click_news_index_batch, dtype=np.int64
        )
        res["candidate_news_entity_index_batch"] = np.asarray(
            candidate_news_entity_index_batch, dtype=np.int64
        )
        res["click_news_entity_index_batch"] = np.asarray(
            click_news_entity_index_batch, dtype=np.int64
        )
        res["impression_id"] = np.asarray(impression_id_list, dtype=np.int64)
        return res

    def _convert_infer_data(
        self,
        candidate_news_index_batch,
        candidate_news_entity_index_batch,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            candidate_news_index_batch (list): the candidate news article's words indices
            candidate_news_entity_index_batch (list): the candidate news article's entities indices
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        res = {}
        res["candidate_news_index_batch"] = np.asarray(
            candidate_news_index_batch, dtype=np.int64
        )
        res["candidate_news_entity_index_batch"] = np.asarray(
            candidate_news_entity_index_batch, dtype=np.int64
        )
        return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        feed_dict = {
            self.labels: data_dict["labels"].reshape([-1, 1]),
            self.candidate_news_index_batch: data_dict[
                "candidate_news_index_batch"
            ].reshape([self.batch_size, self.doc_size]),
            self.click_news_index_batch: data_dict[
                "click_news_index_batch"
            ].reshape([self.batch_size, self.his_size, self.doc_size]),
            self.candidate_news_entity_index_batch: data_dict[
                "candidate_news_entity_index_batch"
            ].reshape([-1, self.doc_size]),
            self.click_news_entity_index_batch: data_dict[
                "click_news_entity_index_batch"
            ].reshape([-1, self.his_size, self.doc_size])

        }
        return feed_dict

    def gen_infer_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.

                Args:
                    data_dict (dict): a dictionary that maps string name to numpy arrays.

                Returns:
                    dict: a dictionary that maps graph elements to numpy arrays.

                """
        feed_dict = {
            self.candidate_news_index_batch: data_dict[
                "candidate_news_index_batch"
            ].reshape([self.batch_size, self.doc_size]),
            self.candidate_news_entity_index_batch: data_dict[
                "candidate_news_entity_index_batch"
            ].reshape([-1, self.doc_size]),
        }
        return feed_dict

