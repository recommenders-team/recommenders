# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

from reco_utils.recommender.deeprec.io.dkn_iterator import DKNTextIterator

r"""

 """


class DKNItem2itemTextIterator(DKNTextIterator):
    def __init__(self, hparams, graph):
        """This new iterator is for DKN's item-to-item recommendations version.
        The tutorial can be found `on this notebook <https://github.com/microsoft/recommenders/blob/main/examples/07_tutorials/KDD2020-tutorial/step4_run_dkn_item2item.ipynb>`.

        Compared with user-to-item recommendations, we don't need the user behavior module.
        So the placeholder can be simplified from the original DKNTextIterator.

        Args:
            hparams (obj): Global hyper-parameters.
            graph (obj): The running graph.
        """
        self.hparams = hparams
        self.graph = graph
        self.neg_num = hparams.neg_num
        self.batch_size = hparams.batch_size * (self.neg_num + 2)
        self.doc_size = hparams.doc_size
        with self.graph.as_default():
            self.candidate_news_index_batch = tf.placeholder(
                tf.int64, [self.batch_size, self.doc_size], name="candidate_news_index"
            )
            self.candidate_news_entity_index_batch = tf.placeholder(
                tf.int64,
                [self.batch_size, self.doc_size],
                name="candidate_news_entity_index",
            )

        self._loading_nessary_files()

    def _loading_nessary_files(self):
        """Only one feature file is needed:  `news_feature_file`
        This function loads the news article's features into two dictionaries: `self.news_word_index` and `self.news_entity_index`.
        """
        hparams = self.hparams
        self.news_word_index = {}
        self.news_entity_index = {}
        with open(hparams.news_feature_file, "r") as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                newsid, word_index, entity_index = line.strip().split(" ")
                self.news_word_index[newsid] = [
                    int(item) for item in word_index.split(",")
                ]
                self.news_entity_index[newsid] = [
                    int(item) for item in entity_index.split(",")
                ]

    def load_data_from_file(self, infile):
        """This function will return a mini-batch of data with features,
        by looking up `news_word_index` dictionary and `news_entity_index` dictionary according to the news article's ID.

        Args:
            infile (str): File path. Each line of `infile` is a news article's ID.

        Yields:
            dict, list, int:
            - A dictionary that maps graph elements to numpy arrays.
            - A list with news article's ID.
            - Size of the data in a batch.

        """
        newsid_list = []
        candidate_news_index_batch = []
        candidate_news_entity_index_batch = []
        cnt = 0
        with open(infile, "r") as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                newsid = line.strip()
                word_index, entity_index = (
                    self.news_word_index[newsid],
                    self.news_entity_index[newsid],
                )
                newsid_list.append(newsid)

                candidate_news_index_batch.append(word_index)
                candidate_news_entity_index_batch.append(entity_index)

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
