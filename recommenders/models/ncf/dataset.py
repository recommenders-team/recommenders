# Read from disk, sample in memory, scan dataset with ordereddict of users and file locations
# Original code samples training examples WITH REPLACEMENT (because it samples repetitively for each positive example). However in paper it samples without replacement,
# 4 negative examples for each positive example. For training samples, the original code samples 100 negatives for each positive example which corresponds to the paper (although the paper
# only has one positive example for each user in the test set)
# Somehow original code can generate predictions for test items that did not appear in training set. It does this because embeddings are created for all items across both training and test. All users appear in both sets so it is unclear whether the same is try for users.
# Original code can sample from items that only appear in test set when generating negative samples in training. This could be seen as unrealistic as it does not match a real world situation
# where the test set is completely unseen.

from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import warnings
import math
import csv

from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


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
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        binary=True,
        seed=None,
        sample_with_replacement=False,
        print_warnings=True
    ):
        """Constructor
        Args:
            train_file (str): Path to training dataset file.
            test_file (str): Path to test dataset file.
            test_file_full (str): Path to full test dataset file including negative samples.
            overwrite_test_file_full (bool): If true, recreate and overwrite test_file_full.
            n_neg (int): Number of negative samples per positive example for training set.
            n_neg_test (int): Number of negative samples per positive example for test set.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            col_timestamp (str): Timestamp column name.
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
        self.col_timestamp = col_timestamp
        self.binary = binary
        self.sample_with_replacement = sample_with_replacement
        self.print_warnings = print_warnings

        self.col_test_batch = "test_batch"
        
        # set sampling method to use
        if self.sample_with_replacement:
            self._sample = self._sample_negatives_with_replacement
        else:
            self._sample = self._sample_negatives_without_replacement

        # data preprocessing for training and test data
        (
            self.train_user_pool, self.train_item_pool, self.train_len,
            self.user2id, self.item2id
        ) = self._process_dataset(self.train_file, train_set=True)
        
        self.n_users = len(self.train_user_pool)
        self.n_items = len(self.train_item_pool)
            
        if test_file:
            if self.test_file_full is None:
                self.test_file_full = os.path.splitext(self.test_file)[0] + "_full.csv"
            self.test_user_pool, self._test_item_pool, self.test_len, _, _ = self._process_dataset(
                self.test_file, train_set=False
            )
            if os.path.isfile(self.test_file_full):
                self._validate_test_file()
            else:
                self._create_test_set()
            
        self.id2item = {self.item2id[k]: k for k in self.item2id}
        self.id2user = {self.user2id[k]: k for k in self.user2id}
        
        # set random seed
        random.seed(seed)
    

    def _check_for_missing_fields(self, fields_to_check, filename, reader):
        missing_fields = set([fields_to_check).difference(set(reader.fieldnames))
        if len(missing_fields):
            raise ValueError("Columns {} not in header of file {}".format(missing_fields, filename))


    def _extract_row_data(self, row):
        user = int(row[self.col_user])
        item = int(row[self.col_item])
        rating = float(row[self.col_rating])
        if self.binary:
            rating = float(rating > 0)
        surplus_data = {k:v for k, v in row.items() if k not in [self.col_user, self.col_item, self.col_rating]}
        return user, item, rating, surplus_data


    def _process_dataset(self, filename, train_set=True):
        # Scan dataset, record the users present and line location in file, record the items present in file,
        # and validate that the dataset is sorted by user.
                
        with open(filename, 'r', encoding='UTF8') as f:
            
            reader = csv.DictReader(f)
            
            self._check_for_missing_fields([self.col_user, self.col_item, self.col_rating], reader, filename)
            
            row = next(reader, None)
            if not row:
                raise("File {} is empty".format(filename))
            user, item, rating, _ = self._extract_row_data(row)
            current_user = user
            current_user_items = []
            user_line_start = reader.line_num # starts at 2
            user_pool = OrderedDict()
            item_pool = [item]
            item2id, user2id = {item: 0}, {user: 0}
            
            while row:
                row = next(reader, None)
                if row:
                    user, item, rating, _ = self._extract_row_data(row)
                
                    if item not in item_pool:
                        item_pool.append(item)
                        item2id[item] = len(item2id)

                    current_user_items.append(item)
                
                if (user != current_user) or not row:
                    user_line_end = reader.line_num - 1
                    
                    # if last line of file
                    if not row:
                        current_user = user
                        user_line_end = reader.line_num
                        
                    if current_user in user_pool:
                        dataset_name = "training" if training_set else "test"
                        raise ValueError("{} dataset is not sorted by user".format(dataset_name))

                    user_info = {
                        'user_line_start': user_line_start, 'user_line_end': user_line_end
                    }
                    user_pool[current_user] = user_info
                    user2id[current_user] = len(user2id)
                    
                    current_user = user
                    current_user_items = []
                    user_line_start = reader.line_num
                    
            data_len = reader.line_num - 1
    
        return user_pool, item_pool, data_len, user2id, item2id


    def _validate_test_file(self):
        current_row_batch = 0
        with open(self.test_file_full, 'r', encoding='UTF8') as f:
            reader = csv.DictReader(f)

            row = next(reader, None)
            if not row:
                raise("File {} is empty".format(self.test_file_full))

            self._check_for_missing_fields(
                [self.col_user, self.col_item, self.col_rating, self.col_test_batch],
                reader, self.test_file_full
            )

            user, item, rating, surplus_data = self._extract_row_data(row)
            current_user = user
            while row:
                row = next(reader, None)
                if row:
                    user, item, rating, surplus_data = self._extract_row_data(row)
                    row_batch = int(surplus_data[self.col_test_batch])
                    if user not in self.train_user_pool:
                        raise ValueError("User {} in test set but not in training set".format(user))
                    if user not in self.train_item_pool:
                        raise ValueError("Item {} in test set but not in training set".format(item))
                    if (row_batch != current_row_batch) or not row:
                        if not row_batch == current_row_batch + 1:
                            raise ValueError("Test file full not sorted by batch number")
                        current_row_batch = row_batch


    def _sample_negatives_with_replacement(self, user_negative_item_pool, n_samples):
        return random.choices(user_negative_item_pool, k=n_samples)


    def _sample_negatives_without_replacement(self, user_negative_item_pool, n_samples):
        return random.sample(user_negative_item_pool, k=n_samples)


    def _get_user_negatives_pool(self, user_positive_item_pool):
        # get list of items user has not interacted with
        return list(set(self.train_item_pool) - user_positive_item_pool)


    def _get_negative_examples(self, user, user_negative_item_pool, n_samples):
        # create dataframe containing negative examples for user assigned zero rating
        user_negative_samples = self._sample(user_negative_item_pool, n_samples)
        return pd.DataFrame({
            self.col_user: [user] * n_samples,
            self.col_item: user_negative_samples,
            self.col_rating: [0.0] * n_samples
        })


    def _load_user_data(self, user_line_indices, reader):
        # load user data from file according to line location and return a dataframe
        user_line_start, user_line_end = user_line_indices["user_line_start"], user_line_indices["user_line_end"]
        user_records = []
        while reader.line_num < user_line_end:
            row = next(reader)
            if reader.line_num >= user_line_start:
                user, item, rating, _ = self._extract_row_data(row)
                user_records.append({self.col_user: user, self.col_item: item, self.col_rating: rating})
            
        return pd.DataFrame.from_records(user_records)
            
    
    def _check_sample_size(self, n_samples, population_size, user, max_n_neg, train_set=True):
        if n_samples > population_size:
            n_samples = min(n_samples, population_size)
            if self.print_warnings:
                n_neg_var = "n_neg" if train_set else "n_neg_test"
                dataset_name = "training" if train_set else "test"
                warnings.warn(
                    "The population of negative items to sample from is too small for user {}. \
                    Reducing {} for this user to {}. \
                    The number of negative samples in the {} set will not be equal for all users. \
                    If an equal number of negative samples for each user is required, \
                    sample with replacement or reduce {}. \
                    This warning can be turned off by setting print_warnings=False" \
                    .format(user, n_neg_var, max_n_neg, dataset_name, n_neg_var)
                )
        
        return n_samples
        

    def _create_test_set(self):
    
        # create empty test_file_full
        pd.DataFrame(
            columns=[self.col_user, self.col_item, self.col_rating, self.col_test_batch]
        ).to_csv(self.test_file_full, index=False)
            
        batch_idx = 0

        with open(self.train_file, 'r', encoding='UTF8') as train_f:
            with open(self.test_file, 'r', encoding='UTF8') as test_f:
                train_reader, test_reader = csv.DictReader(train_f), csv.DictReader(test_f) 

                # for each user, sample n_neg_test negatives for each positive example
                for user in self.test_user_pool.keys():
                    if user in self.train_user_pool:
                        user_test_data = self._load_user_data(self.test_user_pool[user], test_reader)
                        user_train_data = self._load_user_data(self.train_user_pool[user], train_reader)
                        user_positive_item_pool = set(
                            user_test_data[self.col_item].unique()).union(user_train_data[self.col_item].unique()
                        )
                        user_negative_item_pool = self._get_user_negatives_pool(user_positive_item_pool)
                        n_samples = self.n_neg_test

                        # reduce n_samples if sample pool is not large enough
                        if not self.sample_with_replacement:
                            population_size = len(user_negative_item_pool)
                            max_n_neg_test = population_size
                            n_samples = self._check_sample_size(
                                n_samples, population_size, user, max_n_neg_test, train_set=False
                            )
                        
                        user_examples_dfs = []
                        for positive_example in np.array_split(user_test_data, user_test_data.shape[0]):
                            negative_examples = self._get_negative_examples(user, user_negative_item_pool, n_samples)
                            examples = pd.concat([positive_example, negative_examples])
                            examples[self.col_test_batch] = batch_idx
                            user_examples_dfs.append(examples)
                            batch_idx += 1
                            
                        user_examples = pd.concat(user_examples_dfs)
                        
                        user_examples.to_csv(self.test_file_full, mode='a', index=False, header=False)
    

    def _split_into_batches(self, shuffle_buffer, batch_size):
        for i in range(0, len(shuffle_buffer), batch_size): 
            yield shuffle_buffer[i:i + batch_size]


    def _prepare_batch_with_id(self, batch):
        return [
            [self.user2id[user] for user in batch[self.col_user].values],
            [self.item2id[item] for item in batch[self.col_item].values],
            batch[self.col_rating].values
        ]
    

    def _prepare_batch_without_id(self, batch):
        return [
            batch[self.col_user].values.tolist(),
            batch[self.col_item].values.tolist(),
            batch[self.col_rating].values.tolist()
        ]


    def _release_shuffle_buffer(self, shuffle_buffer, batch_size, yield_id, write_to = None):
        prepare_batch = self._prepare_batch_with_id if yield_id else self._prepare_batch_without_id
        shuffle_buffer_df = pd.concat(shuffle_buffer)
        shuffle_buffer_df = shuffle_buffer_df.sample(shuffle_buffer_df.shape[0])
        for batch in self._split_into_batches(shuffle_buffer_df, batch_size):
            if batch.shape[0] == batch_size:
                if write_to:
                    batch.to_csv(write_to, mode='a', header=False, index=False)
                yield prepare_batch(batch)
            else:
                return batch


    def train_loader(self, batch_size, shuffle_size=None, yield_id=True, write_to=None):
        """
        Generator for serving batches of training data. Positive examples are loaded from the
        original training file, to which negative samples are added. Data is loaded in memory into a
        shuffle buffer up to a maximum of shuffle_size rows, before the data is shuffled and released.
        If out-of-memory errors are encountered, try reducing shuffle_size.

        Args:
            batch_size (int): Number of examples in each batch.
            shuffle_size (int): Maximum Number of examples in shuffle buffer.
            yield_id (bool): If true, return assigned user and item IDs, else return original values.
            write_to (str): Path of file to write full dataset (including negative examples) to. 
        """

        # if shuffle_size not supplied, use (estimated) full data size i.e. complete in-memory shuffle
        if shuffle_size is None:
            shuffle_size = (self.train_len * (self.n_neg + 1))
        
        if write_to:
            pd.DataFrame(columns=[self.col_user, self.col_item, self.col_rating]).to_csv(write_to, header=True, index=False)
        
        shuffle_buffer = []
        
        with open(self.train_file, 'r', encoding='UTF8') as f:
            reader = csv.DictReader(f)
            for user, file_indices in self.train_user_pool.items():
                user_positive_examples = self._load_user_data(file_indices, reader)
                user_positive_item_pool = set(user_positive_examples[self.col_item].unique())
                user_negative_item_pool = self._get_user_negatives_pool(user_positive_item_pool)
                n_samples = self.n_neg * user_positive_examples.shape[0]

                # reduce n_samples if sample pool is not large enough
                if not self.sample_with_replacement:
                    population_size = len(user_negative_item_pool)
                    max_n_neg = population_size // user_positive_examples.shape[0]
                    n_samples = self._check_sample_size(
                        n_samples, population_size, user, max_n_neg, train_set=True
                    )

                user_negative_examples = self._get_negative_examples(user, user_positive_item_pool, n_samples)
                user_examples = pd.concat([user_positive_examples, user_negative_examples])
                shuffle_buffer.append(user_examples)
                shuffle_buffer_len = sum([df.shape[0] for df in shuffle_buffer])
                if (shuffle_buffer_len >= shuffle_size):
                    buffer_remainder = yield from self._release_shuffle_buffer(shuffle_buffer, batch_size, yield_id, write_to)
                    shuffle_buffer = [buffer_remainder] if buffer_remainder is not None else []
            
            # yield remaining buffer
            yield from self._release_shuffle_buffer(shuffle_buffer, batch_size, yield_id, write_to)
                    

    def test_loader(self, yield_id=True):
        """
        Generator for serving batches of test data. Data is loaded from test_file_full file.

        Args:
            yield_id (bool): If true, return assigned user and item IDs, else return original values.
        """
        prepare_batch = self._prepare_batch_with_id if yield_id else self._prepare_batch_without_id
            
        with open(self.test_file_full, 'r', encoding='UTF8') as f:
            reader = csv.DictReader(f)
            batch_users, batch_items, batch_ratings = [], [], []
            batch_idx = 0
            for row in reader:
                user, item, rating, surplus_data = self._extract_row_data(row)
                row_batch = int(surplus_data[self.col_test_batch])
                if row_batch != batch_idx:
                    yield prepare_batch(
                        pd.DataFrame(
                                {
                                    self.col_user: batch_users,
                                    self.col_item: batch_items,
                                    self.col_rating: batch_ratings
                                }
                        )
                    )
                    batch_users, batch_items, batch_ratings = [], [], []
                    batch_idx += 1
                batch_users.append(user)
                batch_items.append(item)
                batch_ratings.append(rating)
            
            # yield final batch
            yield prepare_batch(
                pd.DataFrame(
                        {
                            self.col_user: batch_users,
                            self.col_item: batch_items,
                            self.col_rating: batch_ratings
                        }
                )
            )
