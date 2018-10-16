"""
Collection of Splitter Methods
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
# pylint: disable=E0611
import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number, broadcast
from pyspark.sql.functions import round as spark_round
from sklearn.model_selection import train_test_split as sk_split
import surprise
from surprise.model_selection import train_test_split

from utilities.datasets.surprise_dataset import SurprisePandasDataset
from utilities.recommenders.recommender_base import DEFAULT_USER_COL, DEFAULT_RATING_COL, \
    DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL


class SplittersBase(ABC):
    """
    Splitter Interface
    """
    def __init__(self, ratio=0.75, seed=123):
        """Splitter interface
        Args:
            ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into
            two halfs and the ratio argument indicates the ratio of training data set;
            if it is a list of float numbers, the splitter splits data into several portions
            corresponding to the
            split ratios. If a list is provided and the ratios are not summed to 1, they will be
            normalized.
            seed (int): Seed.
        """
        if isinstance(ratio, float):
            if ratio <= 0 or ratio >= 1:
                raise ValueError("Split ratio has to be between 0 and 1")
            self.multi_split = False
        elif isinstance(ratio, list):
            if any([x <= 0 for x in ratio]):
                raise ValueError(
                    "All split ratios in the ratio list should be larger than 0.")

            # normalize split ratios if they are not summed to 1
            if sum(ratio) != 1.0:
                ratio = [x / sum(ratio) for x in ratio]

            self.multi_split = True
        else:
            raise TypeError("Split ratio should be either float or a list of floats.")

        self._ratio = ratio
        self.seed = seed

    def split(self, data):
        """Main Split Function
        Args:
            data: data to be split
        Returns:
            split for each ratio value provided in the same format as the input data

        """
        if isinstance(data, surprise.Dataset):
            return self._surprise_split(data)
        if isinstance(data, surprise.Trainset):
            return self._surprise_split_trainset(data)
        if isinstance(data, pd.DataFrame):
            return self._pandas_split(data)
        try:
            import pyspark
            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                return self._spark_split(data)
        except NameError:
            raise NotImplementedError("Spark not installed")
        raise NotImplementedError("Unsupported Split Type")

    def _surprise_split_trainset(self, data):
        """Surprise Splitter Interface Method
        Args:
            data (surprise.Trainset): Surprise dataset to be split.
        """
        schema = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        dataset = data.all_ratings()
        dataframe = pd.DataFrame(dataset,
                                 columns=schema)
        return self._surprise_split(
            SurprisePandasDataset(schema=schema).load_dataset(dataframe=dataframe))

    @abstractmethod
    def _surprise_split(self, data, **kwargs):
        """Surprise Splitter Interface Method
        Args:
            data (surprise.Dataset): Surprise dataset to be split.
        """
        pass

    @abstractmethod
    def _spark_split(self, data, **kwargs):
        """Spark Splitter Interface Method
        Args:
            data (pyspark.DataFrame): Spark dataframe to be split.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _pandas_split(self, data, **kwargs):
        """Pandas Splitter Interface Method
        Args:
            data (pandas.DataFrame): pandas dataframe to be split.
        """
        pass  # pragma: no cover

    @property
    def ratio(self):
        """Split ratio of the dataset"""
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        """Provide decimal to split data into left set, and the rest returned in right set"""
        self._ratio = ratio


class RandomSplitter(SplittersBase):
    """Random Splitter"""

    def _surprise_split(self, data, **kwargs):

        if self.multi_split:
            indices = np.arange(len(data.raw_ratings))

            np.random.seed(self.seed)
            np.random.shuffle(indices)

            split_index = []
            for index, _ in enumerate(self.ratio[:-1]):
                split_index.append(sum(self.ratio[:index + 1]))

            indices_splits = np.split(indices,
                                      [int(x * len(indices)) for x in split_index])
            splits = []
            for indices_split in indices_splits:
                data_split = [data.raw_ratings[i] for i in indices_split]

                # The returned splits are lists of user-item-ratings tuples.
                surprise_data_split = data.construct_testset(data_split)

                splits.append(surprise_data_split)

            return splits
        else:
            return train_test_split(data, test_size=None, train_size=self.ratio, random_state=self.seed)

    def _spark_split(self, data, **kwargs):
        if self.multi_split:
            return data.randomSplit(self.ratio, seed=self.seed)
        else:
            return data.randomSplit([self.ratio, 1 - self.ratio],
                                    seed=self.seed)

    def _pandas_split(self, data, **kwargs):
        if self.multi_split:
            splits = _split_pandas_data_with_ratios(data, self.ratio,
                                                    resample=True, seed=self.seed)
            return splits
        else:
            return sk_split(data, test_size=None,
                            train_size=self.ratio, random_state=self.seed)


class ChronoSplitter(SplittersBase):
    """Chronological Splitter

    Chronological splitting split data (items are ordered by timestamps for each customer) by
    timestamps.
    """

    def __init__(self, ratio=0.75, seed=123, min_rating=1, filter_by="user",
                 col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
        """
        Args:
            ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into
            two halfs and the ratio argument indicates the ratio of training data set;
            if it is a list of float numbers, the splitter splits data into several portions
            corresponding to the
            split ratios. If a list is provided and the ratios are not summed to 1, they will be
            normalized.
            seed (int): Seed.
            min_rating (int): minimum number of ratings for user or item.
            filter_by (str): either "user" or "item", depending on which of the two is to filter
            with
            min_rating.
            col_user (str): column name of user IDs.
            col_item (str): column name of item IDs.
            col_timestamp (str): column name of timestamps.
        """
        if not (filter_by == "user" or filter_by == "item"):
            raise ValueError("filter_by should be either 'user' or 'item'.")

        if min_rating < 1:
            raise ValueError("min_rating should be integer and larger than or equal to 1.")

        super().__init__(ratio=ratio, seed=seed)
        self.col_timestamp = col_timestamp
        self.col_item = col_item
        self.col_user = col_user
        self.col_rating = col_rating
        self.filter_by = filter_by
        self.min_rating = min_rating

    def _surprise_split(self, data, **kwargs):
        raise NotImplementedError("Surprise version has not yet implemented.")

    def _spark_split(self, data, **kwargs):
        split_by_column = self.col_user if self.filter_by == "user" else self.col_item

        if self.min_rating > 1:
            data = min_rating_filter(data, min_rating=self.min_rating, filter_by=self.filter_by,
                                     col_user=self.col_user, col_item=self.col_item)

        ratio = self.ratio if self.multi_split else [self.ratio, 1 - self.ratio]
        ratio_index = np.cumsum(ratio)

        window_spec = Window.partitionBy(
            split_by_column).orderBy(col(self.col_timestamp).desc())

        rating_grouped = data \
            .groupBy(split_by_column) \
            .agg({self.col_timestamp: 'count'}) \
            .withColumnRenamed('count(' + self.col_timestamp + ')', 'count')
        rating_all = data.join(broadcast(rating_grouped), on=split_by_column)

        rating_rank = rating_all \
            .withColumn('rank', row_number().over(window_spec) / col('count'))

        splits = []
        for i, _ in enumerate(ratio_index):
            if i == 0:
                rating_split = rating_rank \
                    .filter(col('rank') <= ratio_index[i])
            else:
                rating_split = rating_rank \
                    .filter((col('rank') <= ratio_index[i]) & (col('rank') > ratio_index[i-1]))

            splits.append(rating_split)

        return splits

    def _pandas_split(self, data, **kwargs):
        split_by_column = self.col_user if self.filter_by == "user" else self.col_item

        # Sort data by timestamp.
        data = data.sort_values(by=[split_by_column, self.col_timestamp],
                                axis=0,
                                ascending=False)

        ratio = self.ratio if self.multi_split else [self.ratio, 1 - self.ratio]

        if self.min_rating > 1:
            data = min_rating_filter(data, min_rating=self.min_rating,
                                     filter_by=self.filter_by,
                                     col_user=self.col_user, col_item=self.col_item)

        num_of_splits = len(ratio)
        splits = [pd.DataFrame({})] * num_of_splits
        df_grouped = data \
            .sort_values(self.col_timestamp) \
            .groupby(split_by_column)
        for name, group in df_grouped:
            group_splits = _split_pandas_data_with_ratios(df_grouped.get_group(name), ratio,
                                                          resample=False)
            for x in range(num_of_splits):
                splits[x] = pd.concat([splits[x], group_splits[x]])

        return splits


def min_rating_filter(data, min_rating=1, filter_by="user",
                      col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a
    new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is
    called warm if he has rated at least 4 items.

    Args:
        data (spark.DataFrame or pandas.DataFrame): Spark data frame or Pandas data frame of
        user-item tuples. Columns of user and item should be
        present in the data frame while other columns like rating, timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to filter with
        min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        Spark DataFrame with at least columns of user and item that has been filtered by the
        given specifications.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user

    if isinstance(data, pd.DataFrame):
        rating_filtered = data.groupby(split_by_column) \
            .filter(lambda x: len(x) >= min_rating)

        return rating_filtered
    try:
        import pyspark
        from pyspark.sql.functions import col, broadcast
        if isinstance(data, pyspark.sql.DataFrame):
            rating_temp = data.groupBy(split_by_column) \
                .agg({split_with_column: "count"}) \
                .withColumnRenamed('count(' + split_with_column + ')',
                                   "n" + split_with_column) \
                .where(col("n" + split_with_column) >= min_rating)

            rating_filtered = data.join(broadcast(rating_temp), split_by_column) \
                .drop("n" + split_with_column)

            return rating_filtered
    except NameError:
        raise TypeError("Spark not installed")
    raise TypeError("Only Spark and Pandas Data Frames are supported for min rating filter.")



def _split_pandas_data_with_ratios(data, ratios, seed=1234, resample=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train
        -validation-and-test

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split.
        seed (int): random seed.
        resample (bool): whether data will be resampled when being split.

    Returns:
        List of data frames which are split by the given specifications.
    """
    split_index = np.cumsum(ratios).tolist()[:-1]

    if resample:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data,
                      [round(x * len(data)) for x in split_index])

    return splits

