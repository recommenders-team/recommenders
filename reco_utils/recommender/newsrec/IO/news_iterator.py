import tensorflow as tf
import numpy as np

from reco_utils.recommender.deeprec.IO.iterator import BaseIterator

__all__ = ["NewsTrainIterator", "NewsTestIterator"]

class NewsTrainIterator(BaseIterator):
    """Train data loader for the NRMS NPA LSTUR model.
    Those model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articlesand user's clicked news article. Articles are represented by title words. 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        doc_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
    """

    def __init__(self, hparams, col_spliter=" ", ID_spliter="%"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.doc_size = hparams.doc_size
        self.his_size = hparams.his_size
        self.npratio = hparams.npratio

    
    def parser_one_line(self, line):
        """Parse one string line into feature values.
        
        Args:
            line (str): a string indicating one instance

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_news_index, clicked_news_index.
        """
        words = line.strip().split(self.ID_spliter)

        cols = words[0].strip().split(self.col_spliter)
        label = [float(i) for i in cols[:self.npratio+1]]
        candidate_news_index = []
        click_news_index = []
        imp_index = []
        user_index = []

        for news in cols[self.npratio+1:]:
            tokens = news.split(":")
            if "Impression" in tokens[0]:
                imp_index.append(int(tokens[1]))
            elif "User" in tokens[0]:
                user_index.append(int(tokens[1]))
            elif "CandidateNews" in tokens[0]:
                # word index start by 0
                candidate_news_index.append([int(i) for i in tokens[1].split(",")])
            elif "ClickedNews" in tokens[0]:
                click_news_index.append([int(i) for i in tokens[1].split(",")])
            else:
                raise ValueError("data format is wrong")

        return (
            label,
            imp_index,
            user_index,
            candidate_news_index,
            click_news_index
        )


    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        click_news_indexes = []
        cnt = 0

        with tf.gfile.GFile(infile, "r") as rd:
            while True:
                line = rd.readline()
                if not line:
                    break

                label, imp_index, user_index, candidate_news_index, click_news_index = self.parser_one_line(line)

                candidate_news_indexes.append(candidate_news_index)
                click_news_indexes.append(click_news_index)
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list, 
                        imp_indexes,
                        user_indexes,
                        candidate_news_indexes, 
                        click_news_indexes)
                    candidate_news_indexes = []
                    click_news_indexes = []
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    cnt = 0

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_news_indexes,
        click_news_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_news_indexes (list): the candidate news article's words indices
            click_news_indexes (list): words indices for user's clicked news articles
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
    
        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_news_index_batch = np.asarray(
            candidate_news_indexes, dtype=np.int64
        )
        click_news_index_batch = np.asarray(click_news_indexes, dtype=np.int64)
        return [imp_indexes, user_indexes, click_news_index_batch, candidate_news_index_batch], labels


class NewsTestIterator(BaseIterator):
    """Test data loader for the NRMS NPA LSTUR model.
    Those model require a special type of data format, where each instance contains a label, the candidate news article,
    and user's clicked news article. Articles are represented by title words 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        doc_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
    """

    def __init__(self, hparams, col_spliter=" ", ID_spliter="%"):
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

    
    def parser_one_line(self, line):
        """Parse one string line into feature values.
        
        Args:
            line (str): a string indicating one instance

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_news_index, clicked_news_index.

        """
        words = line.strip().split(self.ID_spliter)

        cols = words[0].strip().split(self.col_spliter)
        label = [float(i) for i in cols[:1]]
        candidate_news_index = []
        click_news_index = []
        imp_index = []
        user_index = []

        for news in cols[1:]:
            tokens = news.split(":")
            if "Impression" in tokens[0]:
                imp_index.append(int(tokens[1]))
            elif "User" in tokens[0]:
                user_index.append(int(tokens[1]))
            elif "CandidateNews" in tokens[0]:
                # word index start by 0
                for i in tokens[1].split(","):
                    candidate_news_index.append(int(i))
            elif "ClickedNews" in tokens[0]:
                click_news_index.append([int(i) for i in tokens[1].split(",")])
            else:
                raise ValueError("data format is wrong")

        return (
            label,
            imp_index,
            user_index,
            candidate_news_index,
            click_news_index
        )


    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        click_news_indexes = []
        cnt = 0

        with tf.gfile.GFile(infile, "r") as rd:
            while True:
                line = rd.readline()
                if not line:
                    break

                label, imp_index, user_index, candidate_news_index, click_news_index= self.parser_one_line(line)

                candidate_news_indexes.append(candidate_news_index)
                click_news_indexes.append(click_news_index)
                label_list.append(label)
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list, 
                        imp_indexes,
                        user_indexes,
                        candidate_news_indexes, 
                        click_news_indexes)
                    candidate_news_indexes = []
                    click_news_indexes = []
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    cnt = 0

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_news_indexes,
        click_news_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_news_indexes (list): the candidate news article's words indices
            click_news_indexes (list): words indices for user's clicked news articles
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
    
        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_news_index_batch = np.asarray(
            candidate_news_indexes, dtype=np.int64
        )
        click_news_index_batch = np.asarray(click_news_indexes, dtype=np.int64)
        return [imp_indexes, user_indexes, click_news_index_batch, candidate_news_index_batch], labels
