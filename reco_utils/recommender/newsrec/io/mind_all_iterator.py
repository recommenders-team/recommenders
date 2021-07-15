# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import pickle

from reco_utils.models.deeprec.io.iterator import BaseIterator
from reco_utils.models.newsrec.newsrec_utils import word_tokenize, newsample

__all__ = ["MINDAllIterator"]


class MINDAllIterator(BaseIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts.

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        body_size (int): max word num in news body (abstract used in MIND).
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(
        self,
        hparams,
        npratio=-1,
        col_spliter="\t",
        ID_spliter="%",
    ):
        """Initialize an iterator. Create necessary placeholders for the model.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            graph (object): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        self.body_size = hparams.body_size
        self.his_size = hparams.his_size
        self.npratio = npratio

        self.word_dict = self.load_dict(hparams.wordDict_file)
        self.vert_dict = self.load_dict(hparams.vertDict_file)
        self.subvert_dict = self.load_dict(hparams.subvertDict_file)
        self.uid2index = self.load_dict(hparams.userDict_file)

    def load_dict(self, file_path):
        """Load pickled file

        Args:
            file path (str): File path

        Returns:
            object: pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def init_news(self, news_file):
        """Init news information given news file, such as `news_title_index`, `news_abstract_index`.

        Args:
            news_file: path of news file
        """
        self.nid2index = {}
        news_title = [""]
        news_ab = [""]
        news_vert = [""]
        news_subvert = [""]

        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                ab = word_tokenize(ab)
                news_title.append(title)
                news_ab.append(ab)
                news_vert.append(vert)
                news_subvert.append(subvert)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        self.news_ab_index = np.zeros((len(news_ab), self.body_size), dtype="int32")
        self.news_vert_index = np.zeros((len(news_vert), 1), dtype="int32")
        self.news_subvert_index = np.zeros((len(news_subvert), 1), dtype="int32")

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            ab = news_ab[news_index]
            vert = news_vert[news_index]
            subvert = news_subvert[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]
            for word_index_ab in range(min(self.body_size, len(ab))):
                if ab[word_index_ab] in self.word_dict:
                    self.news_ab_index[news_index, word_index_ab] = self.word_dict[
                        ab[word_index_ab].lower()
                    ]
            if vert in self.vert_dict:
                self.news_vert_index[news_index, 0] = self.vert_dict[vert]
            if subvert in self.subvert_dict:
                self.news_subvert_index[news_index, 0] = self.subvert_dict[subvert]

    def init_behaviors(self, behaviors_file):
        """Init behavior logs given behaviors file.

        Args:
            behaviors_file (str): path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[
                    : self.his_size
                ]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one string line into feature values.

        Args:
            line (str): a string indicating one instance.

        Yields:
            list: Parsed results including label, impression id , user id,
            candidate_title_index, clicked_title_index,
            candidate_ab_index, clicked_ab_index,
            candidate_vert_index, clicked_vert_index,
            candidate_subvert_index, clicked_subvert_index,
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.npratio

                n = newsample(negs, self.npratio)
                candidate_title_index = self.news_title_index[[p] + n]
                candidate_ab_index = self.news_ab_index[[p] + n]
                candidate_vert_index = self.news_vert_index[[p] + n]
                candidate_subvert_index = self.news_subvert_index[[p] + n]
                click_title_index = self.news_title_index[self.histories[line]]
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]
                click_subvert_index = self.news_subvert_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_title_index,
                    click_ab_index,
                    click_vert_index,
                    click_subvert_index,
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                click_title_index = self.news_title_index[self.histories[line]]

                candidate_title_index = self.news_title_index[news]
                candidate_ab_index = self.news_ab_index[news]
                candidate_vert_index = self.news_vert_index[news]
                candidate_subvert_index = self.news_subvert_index[news]
                click_title_index = self.news_title_index[self.histories[line]]
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]
                click_subvert_index = self.news_subvert_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_title_index,
                    click_ab_index,
                    click_vert_index,
                    click_subvert_index,
                )

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from a file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed results, in the format of graph feed_dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []
        click_title_indexes = []
        click_ab_indexes = []
        click_vert_indexes = []
        click_subvert_indexes = []
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                label,
                impr_index,
                user_index,
                candidate_title_index,
                candidate_ab_index,
                candidate_vert_index,
                candidate_subvert_index,
                click_title_index,
                click_ab_index,
                click_vert_index,
                click_subvert_index,
            ) in self.parser_one_line(index):
                candidate_title_indexes.append(candidate_title_index)
                candidate_ab_indexes.append(candidate_ab_index)
                candidate_vert_indexes.append(candidate_vert_index)
                candidate_subvert_indexes.append(candidate_subvert_index)
                click_title_indexes.append(click_title_index)
                click_ab_indexes.append(click_ab_index)
                click_vert_indexes.append(click_vert_index)
                click_subvert_indexes.append(click_subvert_index)
                imp_indexes.append(impr_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        candidate_ab_indexes,
                        candidate_vert_indexes,
                        candidate_subvert_indexes,
                        click_title_indexes,
                        click_ab_indexes,
                        click_vert_indexes,
                        click_subvert_indexes,
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_title_indexes = []
                    candidate_ab_indexes = []
                    candidate_vert_indexes = []
                    candidate_subvert_indexes = []
                    click_title_indexes = []
                    click_ab_indexes = []
                    click_vert_indexes = []
                    click_subvert_indexes = []
                    cnt = 0

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
        click_title_indexes,
        click_ab_indexes,
        click_vert_indexes,
        click_subvert_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            candidate_ab_indexes (list): the candidate news abstarcts' words indices.
            candidate_vert_indexes (list): the candidate news verts' words indices.
            candidate_subvert_indexes (list): the candidate news subverts' indices.
            click_title_indexes (list): words indices for user's clicked news titles.
            click_ab_indexes (list): words indices for user's clicked news abstarcts.
            click_vert_indexes (list): indices for user's clicked news verts.
            click_subvert_indexes (list):indices for user's clicked news subverts.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int64)
        candidate_vert_index_batch = np.asarray(candidate_vert_indexes, dtype=np.int64)
        candidate_subvert_index_batch = np.asarray(
            candidate_subvert_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.asarray(click_vert_indexes, dtype=np.int64)
        click_subvert_index_batch = np.asarray(click_subvert_indexes, dtype=np.int64)
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "candidate_subvert_batch": candidate_subvert_index_batch,
            "labels": labels,
        }

    def load_user_from_file(self, news_file, behavior_file):
        """Read and parse user data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        click_ab_indexes = []
        click_vert_indexes = []
        click_subvert_indexes = []
        cnt = 0

        for index in range(len(self.impr_indexes)):
            click_title_indexes.append(self.news_title_index[self.histories[index]])
            click_ab_indexes.append(self.news_ab_index[self.histories[index]])
            click_vert_indexes.append(self.news_vert_index[self.histories[index]])
            click_subvert_indexes.append(self.news_subvert_index[self.histories[index]])
            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes,
                    impr_indexes,
                    click_title_indexes,
                    click_ab_indexes,
                    click_vert_indexes,
                    click_subvert_indexes,
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                click_ab_indexes = []
                click_vert_indexes = []
                click_subvert_indexes = []

    def _convert_user_data(
        self,
        user_indexes,
        impr_indexes,
        click_title_indexes,
        click_ab_indexes,
        click_vert_indexes,
        click_subvert_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            user_indexes (list): a list of user indexes.
            click_title_indexes (list): words indices for user's clicked news titles.
            click_ab_indexes (list): words indices for user's clicked news abs.
            click_vert_indexes (list): words indices for user's clicked news verts.
            click_subvert_indexes (list): words indices for user's clicked news subverts.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        impr_indexes = np.asarray(impr_indexes, dtype=np.int32)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.asarray(click_vert_indexes, dtype=np.int64)
        click_subvert_index_batch = np.asarray(click_subvert_indexes, dtype=np.int64)

        return {
            "user_index_batch": user_indexes,
            "impr_index_batch": impr_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
        }

    def load_news_from_file(self, news_file):
        """Read and parse user data from news file.

        Args:
            news_file (str): A file contains several informations of news.

        Yields:
            object: An iterator that yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []
        cnt = 0

        for index in range(len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])
            candidate_ab_indexes.append(self.news_ab_index[index])
            candidate_vert_indexes.append(self.news_vert_index[index])
            candidate_subvert_indexes.append(self.news_subvert_index[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_news_data(
                    news_indexes,
                    candidate_title_indexes,
                    candidate_ab_indexes,
                    candidate_vert_indexes,
                    candidate_subvert_indexes,
                )
                news_indexes = []
                candidate_title_indexes = []
                candidate_ab_indexes = []
                candidate_vert_indexes = []
                candidate_subvert_indexes = []

    def _convert_news_data(
        self,
        news_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            news_indexes (list): a list of news indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            candidate_ab_indexes (list): the candidate news abstarcts' words indices.
            candidate_vert_indexes (list): the candidate news verts' words indices.
            candidate_subvert_indexes (list): the candidate news subverts' words indices.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        news_indexes_batch = np.asarray(news_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int32
        )
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int32)
        candidate_vert_index_batch = np.asarray(candidate_vert_indexes, dtype=np.int32)
        candidate_subvert_index_batch = np.asarray(
            candidate_subvert_indexes, dtype=np.int32
        )

        return {
            "news_index_batch": news_indexes_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "candidate_subvert_batch": candidate_subvert_index_batch,
        }

    def load_impression_from_file(self, behaivors_file):
        """Read and parse impression data from behaivors file.

        Args:
            behaivors_file (str): A file contains several informations of behaviros.

        Yields:
            object: An iterator that yields parsed impression data, in the format of dict.
        """

        if not hasattr(self, "histories"):
            self.init_behaviors(behaivors_file)

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )
