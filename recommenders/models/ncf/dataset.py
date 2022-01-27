import os
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import csv
import logging
from tqdm import tqdm

from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFile():
    """
    DataFile class for NCF. Iterator to read data from a csv file.
    Data must be sorted by user. Includes utilities for loading user data from
    to file, formatting it and returning a Pandas dataframe.
    """

    def __init__(
        self, filename, col_user, col_item, col_rating,
        col_test_batch=None, binary=True
    ):
        self.filename = filename
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_test_batch = col_test_batch
        self.expected_fields = [self.col_user, self.col_item, self.col_rating]
        if self.col_test_batch is not None:
            self.expected_fields.append(self.col_test_batch)
        self.binary = binary
        self.user2id, self.item2id, self.id2user, self.id2item = None, None, None, None
        self._init_data()

    @property
    def users(self):
        return self.user2id.keys()

    @property
    def items(self):
        return self.item2id.keys()

    @property
    def end_of_file(self):
        return (self.line_num > 0) and self.next_row is None

    def __iter__(self):
        return self

    def __enter__(self, *args):
        self.file = open(self.filename, 'r', encoding='UTF8')
        self.reader = csv.DictReader(self.file)
        self._check_for_missing_fields(self.expected_fields)
        self.line_num = 0
        self.row, self.next_row = None, None
        return self

    def __exit__(self, *args):
        self.file.close()
        self.reader = None
        self.line_num = 0
        self.row, self.next_row = None, None

    def __next__(self):
        if self.next_row:
            self.row = self.next_row
        elif self.line_num == 0:
            self.row = self._extract_row_data(next(self.reader, None))
            if self.row is None:
                raise Exception("{} is empty.".format(self.filename))
        else:
            raise StopIteration  # end of file

        self.next_row = self._extract_row_data(next(self.reader, None))
        self.line_num += 1

        return self.row

    def _check_for_missing_fields(self, fields_to_check):
        missing_fields = set(fields_to_check).difference(set(self.reader.fieldnames))
        if len(missing_fields):
            raise ValueError("Columns {} not in header of file {}".format(missing_fields, self.filename))

    def _extract_row_data(self, row):
        if row is None:
            return row
        user = int(row[self.col_user])
        item = int(row[self.col_item])
        rating = float(row[self.col_rating])
        if self.binary:
            rating = float(rating > 0)
        test_batch = None
        if self.col_test_batch:
            test_batch = int(row[self.col_test_batch])
        return {self.col_user: user, self.col_item: item, self.col_rating: rating, self.col_test_batch: test_batch}

    def _init_data(self):
        # Compile lists of unique users and items, assign IDs to users and items,
        # and ensure file is sorted by user (and batch index if test set)
        logger.info("Indexing {} ...".format(self.filename))
        with self:
            user_items = []
            self.item2id, self.user2id = OrderedDict(), OrderedDict()
            batch_index = 0
            for _ in self:
                item = self.row[self.col_item]
                user = self.row[self.col_user]
                test_batch = self.row[self.col_test_batch]
                if not self.end_of_file:
                    next_user = self.next_row[self.col_user]
                    next_test_batch = self.next_row[self.col_test_batch]
                if item not in self.items:
                    self.item2id[item] = len(self.item2id)
                user_items.append(item)

                if (next_user != user) or self.next_row is None:
                    if not self.end_of_file:
                        if next_user in self.users:
                            raise ValueError("File {} is not sorted by user".format(self.filename))
                    self.user2id[user] = len(self.user2id)

                if self.col_test_batch:
                    if (next_test_batch != test_batch) or self.next_row is None:
                        if not self.end_of_file:
                            if next_test_batch < batch_index:
                                raise ValueError("File {} is not sorted by {}".format(self.filename, self.col_test_batch))
                        batch_index += 1

            self.batch_indices_range = range(0, batch_index)
            self.data_len = self.line_num - 1

    def load_data(self, key, by_user=True):
        """ Load data for a specified user or test batch

            Args:
                key (int): user or test batch index
                by_user (bool): load data by usr if True, else by test batch

            Returns:
                pandas.DataFrame
        """
        records = []
        key_col = self.col_user if by_user else self.col_test_batch

        # fast forward in file to user/test batch
        while (self.line_num == 0) or (self.row[key_col] != key):
            if self.end_of_file:
                raise Exception("User {} not in file {}".format(key, self.filename))
            next(self)

        # collect user/test batch data
        while self.row[key_col] == key:
            row = self.row
            if self.col_test_batch in row:
                del row[self.col_test_batch]
            records.append(row)
            if not self.end_of_file:
                next(self)
            else:
                break

        return pd.DataFrame.from_records(records)


class NegativeSampler():
    """
    NegativeSampler class for NCF. Samples a subset of negative items from a given population of items.
    """

    def __init__(self, user, n_samples, user_positive_item_pool, item_pool, sample_with_replacement, print_warnings=True, training=True):
        """Constructor

            Args:
                user (str or int): user to be sampled for
                n_samples (int): number of required samples
                user_positive_item_pool (set): set of items with which user has previously interacted
                item_pool (set): set of all items in population
                sample_with_replacement (bool): If true, sample negative examples with replacement,
                    otherwise without replacement.
                print_warnings (bool): If true, prints warnings if sampling without replacement and
                    there are not enough items to sample from to satisfy n_neg or n_neg_test.
                training (bool): set to true if sampling for the training set or false if for the test set
        """
        self.user = user
        self.n_samples = n_samples
        self.user_positive_item_pool = user_positive_item_pool
        self.item_pool = item_pool
        self.sample_with_replacement = sample_with_replacement
        self.print_warnings = print_warnings
        self.training = training

        self.user_negative_item_pool = self._get_user_negatives_pool()
        self.population_size = len(self.user_negative_item_pool)
        self._sample = self._sample_negatives_with_replacement if self.sample_with_replacement else self._sample_negatives_without_replacement
        if not self.sample_with_replacement:
            self._check_sample_size()

    def sample(self):
        """
        Method for sampling uniformly from a population of negative items

            returns: list
        """
        return self._sample()

    def _get_user_negatives_pool(self):
        # get list of items user has not interacted with
        return list(set(self.item_pool) - self.user_positive_item_pool)

    def _sample_negatives_with_replacement(self):
        return random.choices(self.user_negative_item_pool, k=self.n_samples)

    def _sample_negatives_without_replacement(self):
        return random.sample(self.user_negative_item_pool, k=self.n_samples)

    def _check_sample_size(self):
        # if sampling without replacement, check sample population is sufficient and reduce
        # n_samples if not.
        n_neg_var = "n_neg" if self.training else "n_neg_test"
        dataset_name = "training" if self.training else "test"

        k = min(self.n_samples, self.population_size)
        if k < self.n_samples and self.print_warnings:
            warning_string = (
                "The population of negative items to sample from is too small for user {}. "
                "Samples needed = {}, negative items = {}. "
                "Reducing samples to {} for this user."
                "If an equal number of negative samples for each user is required in the {} set, sample with replacement or reduce {}. "
                "This warning can be turned off by setting print_warnings=False"
                .format(self.user, self.n_samples, self.population_size, self.population_size, dataset_name, n_neg_var)
            )
            logging.warning(
                warning_string
            )
        self.n_samples = k


class Dataset(object):
    """Dataset class for NCF"""

    def __init__(
        self,
        train_file,
        test_file=None,
        test_file_full=None,
        overwrite_test_file_full=False,
        n_neg=4,
        n_neg_test=100,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        binary=True,
        seed=None,
        sample_with_replacement=False,
        print_warnings=False
    ):
        """Constructor

            Args:
                train_file (str): Path to training dataset file.
                test_file (str): Path to test dataset file for leave-one-out evaluation.
                test_file_full (str): Path to full test dataset file including negative samples.
                overwrite_test_file_full (bool): If true, recreate and overwrite test_file_full.
                n_neg (int): Number of negative samples per positive example for training set.
                n_neg_test (int): Number of negative samples per positive example for test set.
                col_user (str): User column name.
                col_item (str): Item column name.
                col_rating (str): Rating column name.
                binary (bool): If true, set rating > 0 to rating = 1.
                seed (int): Seed.
                sample_with_replacement (bool): If true, sample negative examples with replacement,
                    otherwise without replacement.
                print_warnings (bool): If true, prints warnings if sampling without replacement and
                    there are not enough items to sample from to satisfy n_neg or n_neg_test.
        """
        self.train_file = train_file
        self.test_file = test_file
        self.test_file_full = test_file_full
        self.overwrite_test_file_full = overwrite_test_file_full
        self.n_neg = n_neg
        self.n_neg_test = n_neg_test
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.binary = binary
        self.sample_with_replacement = sample_with_replacement
        self.print_warnings = print_warnings

        self.col_test_batch = "test_batch"

        self.train_datafile = DataFile(
            filename=self.train_file,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            binary=self.binary
        )

        self.n_users = len(self.train_datafile.users)
        self.n_items = len(self.train_datafile.items)
        self.user2id = self.train_datafile.user2id
        self.item2id = self.train_datafile.item2id
        self.id2user = {self.user2id[k]: k for k in self.user2id}
        self.id2item = {self.item2id[k]: k for k in self.item2id}
        self.train_len = self.train_datafile.data_len

        if self.test_file is not None:
            self.test_datafile = DataFile(
                filename=self.test_file,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_rating,
                binary=self.binary
            )
            if self.test_file_full is None:
                self.test_file_full = os.path.splitext(self.test_file)[0] + "_full.csv"
            if self.overwrite_test_file_full or not os.path.isfile(self.test_file_full):
                self._create_test_file()

            self.test_full_datafile = DataFile(
                filename=self.test_file_full,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_rating,
                col_test_batch=self.col_test_batch,
                binary=self.binary
            )

        # set random seed
        random.seed(seed)

    def _get_negative_examples_df(self, user, user_negative_samples):
        # create dataframe containing negative examples for user assigned zero rating
        n_samples = len(user_negative_samples)
        return pd.DataFrame({
            self.col_user: [user] * n_samples,
            self.col_item: user_negative_samples,
            self.col_rating: [0.0] * n_samples
        })

    def _create_test_file(self):

        logger.info("Creating full leave-one-out test file {} ...".format(self.test_file_full))

        # create empty csv
        pd.DataFrame(
            columns=[self.col_user, self.col_item, self.col_rating, self.col_test_batch]
        ).to_csv(self.test_file_full, index=False)

        batch_idx = 0

        with self.train_datafile as train_datafile:
            with self.test_datafile as test_datafile:
                for user in tqdm(test_datafile.users):
                    if user in train_datafile.users:
                        user_test_data = test_datafile.load_data(user)
                        user_train_data = train_datafile.load_data(user)
                        # for leave-one-out evaluation, exclude items seen in both training and test sets
                        # when sampling negatives
                        user_positive_item_pool = set(user_test_data[self.col_item].unique()) \
                            .union(user_train_data[self.col_item].unique())
                        sampler = NegativeSampler(user, self.n_neg_test, user_positive_item_pool, self.train_datafile.items, self.sample_with_replacement, self.print_warnings, training=False)

                        user_examples_dfs = []
                        # sample n_neg_test negatives for each positive example and assign a batch index
                        for positive_example in np.array_split(user_test_data, user_test_data.shape[0]):
                            negative_examples = self._get_negative_examples_df(user, sampler.sample())
                            examples = pd.concat([positive_example, negative_examples])
                            examples[self.col_test_batch] = batch_idx
                            user_examples_dfs.append(examples)
                            batch_idx += 1

                        # append user test data to file
                        user_examples = pd.concat(user_examples_dfs)
                        user_examples.to_csv(self.test_file_full, mode='a', index=False, header=False)

    def _split_into_batches(self, shuffle_buffer, batch_size):
        for i in range(0, len(shuffle_buffer), batch_size):
            yield shuffle_buffer[i:i + batch_size]

    def _prepare_batch_with_id(self, batch):
        return [
            [self.user2id[user] for user in batch[self.col_user].values],
            [self.item2id[item] for item in batch[self.col_item].values],
            batch[self.col_rating].values.tolist()
        ]

    def _prepare_batch_without_id(self, batch):
        return [
            batch[self.col_user].values.tolist(),
            batch[self.col_item].values.tolist(),
            batch[self.col_rating].values.tolist()
        ]

    def _release_shuffle_buffer(self, shuffle_buffer, batch_size, yield_id, write_to=None):
        prepare_batch = self._prepare_batch_with_id if yield_id else self._prepare_batch_without_id
        shuffle_buffer_df = pd.concat(shuffle_buffer)
        shuffle_buffer_df = shuffle_buffer_df.sample(shuffle_buffer_df.shape[0])  # shuffle the buffer
        for batch in self._split_into_batches(shuffle_buffer_df, batch_size):
            if batch.shape[0] == batch_size:
                if write_to:
                    batch.to_csv(write_to, mode='a', header=False, index=False)
                yield prepare_batch(batch)
            else:
                return batch

    def train_loader(self, batch_size, shuffle_size=None, yield_id=False, write_to=None):
        """
        Generator for serving batches of training data. Positive examples are loaded from the
        original training file, to which negative samples are added. Data is loaded in memory into a
        shuffle buffer up to a maximum of shuffle_size rows, before the data is shuffled and released.
        If out-of-memory errors are encountered, try reducing shuffle_size.

            Args:
                batch_size (int): Number of examples in each batch.
                shuffle_size (int): Maximum number of examples in shuffle buffer.
                yield_id (bool): If true, return assigned user and item IDs, else return original values.
                write_to (str): Path of file to write full dataset (including negative examples).

            Returns:
                list
        """

        # if shuffle_size not supplied, use (estimated) full data size i.e. complete in-memory shuffle
        if shuffle_size is None:
            shuffle_size = (self.train_len * (self.n_neg + 1))

        if write_to:
            pd.DataFrame(columns=[self.col_user, self.col_item, self.col_rating]) \
                .to_csv(write_to, header=True, index=False)

        shuffle_buffer = []

        with self.train_datafile as train_datafile:
            for user in train_datafile.users:
                user_positive_examples = train_datafile.load_data(user)
                user_positive_item_pool = set(user_positive_examples[self.col_item].unique())
                n_samples = self.n_neg * user_positive_examples.shape[0]
                sampler = NegativeSampler(user, n_samples, user_positive_item_pool, self.train_datafile.items, self.sample_with_replacement, self.print_warnings)
                user_negative_examples = self._get_negative_examples_df(user, sampler.sample())
                user_examples = pd.concat([user_positive_examples, user_negative_examples])
                shuffle_buffer.append(user_examples)
                shuffle_buffer_len = sum([df.shape[0] for df in shuffle_buffer])
                if (shuffle_buffer_len >= shuffle_size):
                    buffer_remainder = yield from self._release_shuffle_buffer(
                        shuffle_buffer, batch_size, yield_id, write_to
                    )
                    shuffle_buffer = [buffer_remainder] if buffer_remainder is not None else []

            # yield remaining buffer
            yield from self._release_shuffle_buffer(shuffle_buffer, batch_size, yield_id, write_to)

    def test_loader(self, yield_id=False):
        """
        Generator for serving batches of test data for leave-one-out evaluation. Data is loaded from test_file_full.

            Args:
                yield_id (bool): If true, return assigned user and item IDs, else return original values.

            Returns:
                list
        """
        prepare_batch = self._prepare_batch_with_id if yield_id else self._prepare_batch_without_id

        with self.test_full_datafile as test_full_datafile:
            for test_batch_idx in test_full_datafile.batch_indices_range:
                test_batch_data = test_full_datafile.load_data(test_batch_idx, by_user=False)
                yield prepare_batch(test_batch_data)
