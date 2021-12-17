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
        n_neg=4,
        n_neg_test=100,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        binary=True,
        seed=None,
        train_file=None,
        test_file=None,
        write_test_file=False,
        sample_with_replacement = False,
        print_warnings=True
    ):
        """Constructor
        Args:
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating). test can be None,
                if so, we only process the training data.
            n_neg (int): Number of negative samples for training set.
            n_neg_test (int): Number of negative samples for test set.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            col_timestamp (str): Timestamp column name.
            binary (bool): If true, set rating > 0 to rating = 1.
            seed (int): Seed.
        """
        # set negative sampling for training and test
        self.n_neg = n_neg
        self.n_neg_test = n_neg_test
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self.binary = binary
        self.train_file = train_file
        self.test_file = test_file
        self.write_test_file = write_test_file
        self.sample_with_replacement = sample_with_replacement
        self.print_warnings = print_warnings
        
        self._sample = self._sample_negatives_with_replacement if self.sample_with_replacement else self._sample_negatives_without_replacement

        # data preprocessing for training and test data
        self.train_user_pool, self.train_item_pool, self.train_len, self.user2id, self.item2id = self._process_dataset(self.train_file)
        
        self.n_users = len(self.train_user_pool)
        self.n_items = len(self.train_item_pool)
            
        if test_file:
            if self.write_test_file:
                self.test_file_full = os.path.splitext(self.test_file)[0] + "_full.csv"
            self.test_user_pool, self._test_item_pool, self.test_len, _, _ = self._process_dataset(self.test_file)
            self._create_test_set()
            
        self.id2item = {self.item2id[k]: k for k in self.item2id}
        self.id2user = {self.user2id[k]: k for k in self.user2id}
        
        # set random seed
        random.seed(seed)
    
    def _extract_row_data(self, row):
        user = int(row[self.col_user])
        item = int(row[self.col_item])
        rating = float(row[self.col_rating])
        if self.binary:
            rating = float(rating > 0)
        surplus_data = {k:v for k, v in row.items() if k not in [self.col_user, self.col_item, self.col_rating]}
        return user, item, rating, surplus_data
    
    def _process_dataset(self, filename):
                
        with open(filename, 'r', encoding='UTF8') as f:
            
            reader = csv.DictReader(f)
            
            missing_fields = set([self.col_user, self.col_item, self.col_rating]).difference(set(reader.fieldnames))
            if len(missing_fields):
                raise ValueError("Columns {} not in file header".format(missing_fields))
            
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
                        raise ValueError("DataFrame is not sorted by user")

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
    
    def _sample_negatives_with_replacement(self, user_negatives, n_samples):
        return random.choices(user_negatives, k=n_samples)
    
    def _sample_negatives_without_replacement(self, user_negatives, n_samples):
        n_neg = min(n_samples, len(user_negatives))
        return random.sample(user_negatives, k=n_neg)

    def _sample_negatives(self, user_positive_items, n_samples):
        user_negatives = list(set(self.train_item_pool) - user_positive_items)
        return self._sample(user_negatives, n_samples)
    
    def _get_negative_examples(self, user, user_positive_items, n_samples):
        user_negative_samples = self._sample_negatives(user_positive_items, n_samples)
        return pd.DataFrame({
            self.col_user: [user] * len(user_negative_samples),
            self.col_item: user_negative_samples,
            self.col_rating: [0.0] * len(user_negative_samples)
        })

    def _load_user_data(self, user_line_indices, reader):
        user_line_start, user_line_end = user_line_indices["user_line_start"], user_line_indices["user_line_end"]
        user_records = []
        while reader.line_num < user_line_end:
            row = next(reader)
            if reader.line_num >= user_line_start:
                user, item, rating, _ = self._extract_row_data(row)
                user_records.append({self.col_user: user, self.col_item: item, self.col_rating: rating})
            
        return pd.DataFrame.from_records(user_records)
    
    def _create_test_set(self):
    
        if self.write_test_file:
            pd.DataFrame(
                columns=[self.col_user, self.col_item, self.col_rating, "batch"]
            ).to_csv(self.test_file_full, index=False)
        else:
            user_test_dfs = []
            
        batch_idx = 0

        with open(self.train_file, 'r', encoding='UTF8') as train_f:
            with open(self.test_file, 'r', encoding='UTF8') as test_f:
                train_reader, test_reader = csv.DictReader(train_f), csv.DictReader(test_f) 

                for user in self.test_user_pool.keys():
                    if user in self.train_user_pool:
                        user_test_data = self._load_user_data(self.test_user_pool[user], test_reader)
                        user_train_data = self._load_user_data(self.train_user_pool[user], train_reader)
                        user_positive_items = set(
                            user_test_data[self.col_item].unique()).union(user_train_data[self.col_item].unique()
                        )
                        n_samples = self.n_neg_test
                        if self.print_warnings and not self.sample_with_replacement:
                            max_n_neg_test = self.n_items - len(user_positive_items)
                            if n_samples > max_n_neg_test:
                                warnings.warn("The population of negative items to sample from is too small for user {}. \
    The number of negative samples in the test set will not be equal for all users. Set n_neg_test to a maximum of {} \
    or sample with replacement if an equal number of negative samples for each user is required. \
    This warning can be turned off by setting print_warnings=False".format(user, max_n_neg_test)
                                )
                        user_examples_dfs = []
                        for positive_example in np.array_split(user_test_data, user_test_data.shape[0]):
                            negative_examples = self._get_negative_examples(user, user_positive_items, n_samples)
                            examples = pd.concat([positive_example, negative_examples])
                            examples["batch"] = batch_idx
                            user_examples_dfs.append(examples)
                            batch_idx += 1
                            
                        user_examples = pd.concat(user_examples_dfs)
                        
                        if self.write_test_file:
                            user_examples.to_csv(self.test_file_full, mode='a', index=False, header=False)
                        else:
                            user_test_dfs.append(user_examples)
                if not self.write_test_file:
                    self.test_set_full = pd.concat(user_test_dfs).reset_index(drop=True)
    
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
            
    def train_loader(self, batch_size, shuffle_size=None, yield_id=True, write_to=False):
        
        if shuffle_size is None:
            shuffle_size = (self.train_len * (self.n_neg + 1))
        
        if write_to:
            pd.DataFrame(columns=[self.col_user, self.col_item, self.col_rating]).to_csv(write_to, header=True, index=False)
        
        shuffle_buffer = []
        
        with open(self.train_file, 'r', encoding='UTF8') as f:
            reader = csv.DictReader(f)
            for user, file_indices in self.train_user_pool.items():
                user_positive_examples = self._load_user_data(file_indices, reader)
                user_positive_items = set(user_positive_examples[self.col_item].unique())
                n_samples = self.n_neg * user_positive_examples.shape[0]
                if self.print_warnings and not self.sample_with_replacement:
                    negative_sample_pool_size = self.n_items - len(user_positive_examples)
                    if n_samples > negative_sample_pool_size:
                        max_n_neg = negative_sample_pool_size // user_positive_examples.shape[0]
                        warnings.warn("The population of negative items to sample from is too small for user {}. \
The ratio of positive to negative samples in the train set will not be equal for all users. Set n_neg to a maximum of {} \
or sample with replacement if an equal ratio for each user is required. \
This warning can be turned off by setting print_warnings=False".format(user, max_n_neg)
                                )
                user_negative_examples = self._get_negative_examples(user, user_positive_items, n_samples)
                user_examples = pd.concat([user_positive_examples, user_negative_examples])
                shuffle_buffer.append(user_examples)
                shuffle_buffer_len = sum([df.shape[0] for df in shuffle_buffer])
                if (shuffle_buffer_len >= shuffle_size):
                    buffer_remainder = yield from self._release_shuffle_buffer(shuffle_buffer, batch_size, yield_id, write_to)
                    shuffle_buffer = [buffer_remainder] if buffer_remainder is not None else []
            
            # yield remaining buffer
            yield from self._release_shuffle_buffer(shuffle_buffer, batch_size, yield_id, write_to)
                    
    def test_loader(self, yield_id=True):
        prepare_batch = self._prepare_batch_with_id if yield_id else self._prepare_batch_without_id
        if not self.write_test_file:
            for i, batch in self.test_set_full.groupby("batch"):
                yield prepare_batch(batch)
            
        else:
            
            with open(self.test_file_full, 'r', encoding='UTF8') as f:
                reader = csv.DictReader(f)
                batch_users, batch_items, batch_ratings = [], [], []
                batch_idx = 0
                for row in reader:
                    user, item, rating, surplus_data = self._extract_row_data(row)
                    row_batch = int(surplus_data["batch"])
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
