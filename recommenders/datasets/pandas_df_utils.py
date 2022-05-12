# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import pandas as pd
import numpy as np
from functools import lru_cache, wraps

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
)


logger = logging.getLogger(__name__)


def user_item_pairs(
    user_df,
    item_df,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    user_item_filter_df=None,
    shuffle=True,
    seed=None,
):
    """Get all pairs of users and items data.

    Args:
        user_df (pandas.DataFrame): User data containing unique user ids and maybe their features.
        item_df (pandas.DataFrame): Item data containing unique item ids and maybe their features.
        user_col (str): User id column name.
        item_col (str): Item id column name.
        user_item_filter_df (pd.DataFrame): User-item pairs to be used as a filter.
        shuffle (bool): If True, shuffles the result.
        seed (int): Random seed for shuffle

    Returns:
        pandas.DataFrame: All pairs of user-item from user_df and item_df, excepting the pairs in user_item_filter_df.
    """

    # Get all user-item pairs
    user_df["key"] = 1
    item_df["key"] = 1
    users_items = user_df.merge(item_df, on="key")

    user_df.drop("key", axis=1, inplace=True)
    item_df.drop("key", axis=1, inplace=True)
    users_items.drop("key", axis=1, inplace=True)

    # Filter
    if user_item_filter_df is not None:
        users_items = filter_by(users_items, user_item_filter_df, [user_col, item_col])

    if shuffle:
        users_items = users_items.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )

    return users_items


def filter_by(df, filter_by_df, filter_by_cols):
    """From the input DataFrame `df`, remove the records whose target column `filter_by_cols` values are
    exist in the filter-by DataFrame `filter_by_df`.

    Args:
        df (pandas.DataFrame): Source dataframe.
        filter_by_df (pandas.DataFrame): Filter dataframe.
        filter_by_cols (iterable of str): Filter columns.

    Returns:
        pandas.DataFrame: Dataframe filtered by `filter_by_df` on `filter_by_cols`.

    """

    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(
            filter_by_df.set_index(filter_by_cols).index
        )
    ]


class LibffmConverter:
    """Converts an input dataframe to another dataframe in libffm format. A text file of the converted
    Dataframe is optionally generated.

    .. note::

        The input dataframe is expected to represent the feature data in the following schema:

        .. code-block:: python

            |field-1|field-2|...|field-n|rating|
            |feature-1-1|feature-2-1|...|feature-n-1|1|
            |feature-1-2|feature-2-2|...|feature-n-2|0|
            ...
            |feature-1-i|feature-2-j|...|feature-n-k|0|

        Where
        1. each `field-*` is the column name of the dataframe (column of label/rating is excluded), and
        2. `feature-*-*` can be either a string or a numerical value, representing the categorical variable or
        actual numerical variable of the feature value in the field, respectively.
        3. If there are ordinal variables represented in int types, users should make sure these columns
        are properly converted to string type.

        The above data will be converted to the libffm format by following the convention as explained in
        `this paper <https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`_.

        i.e. `<field_index>:<field_feature_index>:1` or `<field_index>:<field_feature_index>:<field_feature_value>`,
        depending on the data type of the features in the original dataframe.

    Args:
        filepath (str): path to save the converted data.

    Attributes:
        field_count (int): count of field in the libffm format data
        feature_count (int): count of feature in the libffm format data
        filepath (str or None): file path where the output is stored - it can be None or a string

    Examples:
        >>> import pandas as pd
        >>> df_feature = pd.DataFrame({
                'rating': [1, 0, 0, 1, 1],
                'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
                'field2': [3, 4, 5, 6, 7],
                'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
                'field4': ['1', '2', '3', '4', '5']
            })
        >>> converter = LibffmConverter().fit(df_feature, col_rating='rating')
        >>> df_out = converter.transform(df_feature)
        >>> df_out
            rating field1 field2   field3 field4
        0       1  1:1:1  2:4:3  3:5:1.0  4:6:1
        1       0  1:2:1  2:4:4  3:5:2.0  4:7:1
        2       0  1:3:1  2:4:5  3:5:3.0  4:8:1
        3       1  1:3:1  2:4:6  3:5:4.0  4:9:1
        4       1  1:3:1  2:4:7  3:5:5.0  4:10:1
    """

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.col_rating = None
        self.field_names = None
        self.field_count = None
        self.feature_count = None

    def fit(self, df, col_rating=DEFAULT_RATING_COL):
        """Fit the dataframe for libffm format.
        This method does nothing but check the validity of the input columns

        Args:
            df (pandas.DataFrame): input Pandas dataframe.
            col_rating (str): rating of the data.

        Return:
            object: the instance of the converter
        """

        # Check column types.
        types = df.dtypes
        if not all(
            [
                x == object or np.issubdtype(x, np.integer) or x == np.float
                for x in types
            ]
        ):
            raise TypeError("Input columns should be only object and/or numeric types.")

        if col_rating not in df.columns:
            raise TypeError(
                "Column of {} is not in input dataframe columns".format(col_rating)
            )

        self.col_rating = col_rating
        self.field_names = list(df.drop(col_rating, axis=1).columns)

        return self

    def transform(self, df):
        """Tranform an input dataset with the same schema (column names and dtypes) to libffm format
        by using the fitted converter.

        Args:
            df (pandas.DataFrame): input Pandas dataframe.

        Return:
            pandas.DataFrame: Output libffm format dataframe.
        """
        if self.col_rating not in df.columns:
            raise ValueError(
                "Input dataset does not contain the label column {} in the fitting dataset".format(
                    self.col_rating
                )
            )

        if not all([x in df.columns for x in self.field_names]):
            raise ValueError(
                "Not all columns in the input dataset appear in the fitting dataset"
            )

        # Encode field-feature.
        idx = 1
        self.field_feature_dict = {}
        for field in self.field_names:
            for feature in df[field].values:
                # Check whether (field, feature) tuple exists in the dict or not.
                # If not, put them into the key-values of the dict and count the index.
                if (field, feature) not in self.field_feature_dict:
                    self.field_feature_dict[(field, feature)] = idx
                    if df[field].dtype == object:
                        idx += 1
            if df[field].dtype != object:
                idx += 1

        self.field_count = len(self.field_names)
        self.feature_count = idx - 1

        def _convert(field, feature, field_index, field_feature_index_dict):
            field_feature_index = field_feature_index_dict[(field, feature)]
            if isinstance(feature, str):
                feature = 1
            return "{}:{}:{}".format(field_index, field_feature_index, feature)

        for col_index, col in enumerate(self.field_names):
            df[col] = df[col].apply(
                lambda x: _convert(col, x, col_index + 1, self.field_feature_dict)
            )

        # Move rating column to the first.
        column_names = self.field_names[:]
        column_names.insert(0, self.col_rating)
        df = df[column_names]

        if self.filepath is not None:
            np.savetxt(self.filepath, df.values, delimiter=" ", fmt="%s")

        return df

    def fit_transform(self, df, col_rating=DEFAULT_RATING_COL):
        """Do fit and transform in a row

        Args:
            df (pandas.DataFrame): input Pandas dataframe.
            col_rating (str): rating of the data.

        Return:
            pandas.DataFrame: Output libffm format dataframe.
        """
        return self.fit(df, col_rating=col_rating).transform(df)

    def get_params(self):
        """Get parameters (attributes) of the libffm converter

        Return:
            dict: A dictionary that contains parameters field count, feature count, and file path.
        """
        return {
            "field count": self.field_count,
            "feature count": self.feature_count,
            "file path": self.filepath,
        }


def negative_feedback_sampler(
    df,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_label=DEFAULT_LABEL_COL,
    col_feedback="feedback",
    ratio_neg_per_user=1,
    pos_value=1,
    neg_value=0,
    seed=42,
):
    """Utility function to sample negative feedback from user-item interaction dataset.
    This negative sampling function will take the user-item interaction data to create
    binarized feedback, i.e., 1 and 0 indicate positive and negative feedback,
    respectively.

    Negative sampling is used in the literature frequently to generate negative samples
    from a user-item interaction data.

    See for example the `neural collaborative filtering paper <https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf>`_.

    Args:
        df (pandas.DataFrame): input data that contains user-item tuples.
        col_user (str): user id column name.
        col_item (str): item id column name.
        col_label (str): label column name in df.
        col_feedback (str): feedback column name in the returned data frame; it is used for the generated column
            of positive and negative feedback.
        ratio_neg_per_user (int): ratio of negative feedback w.r.t to the number of positive feedback for each user.
            If the samples exceed the number of total possible negative feedback samples, it will be reduced to the
            number of all the possible samples.
        pos_value (float): value of positive feedback.
        neg_value (float): value of negative feedback.
        inplace (bool):
        seed (int): seed for the random state of the sampling function.

    Returns:
        pandas.DataFrame: Data with negative feedback.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
            'userID': [1, 2, 3],
            'itemID': [1, 2, 3],
            'rating': [5, 5, 5]
        })
        >>> df_neg_sampled = negative_feedback_sampler(
            df, col_user='userID', col_item='itemID', ratio_neg_per_user=1
        )
        >>> df_neg_sampled
        userID  itemID  feedback
        1   1   1
        1   2   0
        2   2   1
        2   1   0
        3   3   1
        3   1   0
    """
    # Get all of the users and items.
    items = df[col_item].unique()
    rng = np.random.default_rng(seed=seed)

    def sample_items(user_df):
        # Sample negative items for the data frame restricted to a specific user
        n_u = len(user_df)
        neg_sample_size = max(round(n_u * ratio_neg_per_user), 1)
        # Draw (n_u + neg_sample_size) items and keep neg_sample_size of these
        # that are not already in user_df. This requires a set difference from items_sample
        # instead of items, which is more efficient when len(items) is large.
        sample_size = min(n_u + neg_sample_size, len(items))
        items_sample = rng.choice(items, sample_size, replace=False)
        new_items = np.setdiff1d(items_sample, user_df[col_item])[:neg_sample_size]
        new_df = pd.DataFrame(
            data={
                col_user: user_df.name,
                col_item: new_items,
                col_label: neg_value,
            }
        )
        return pd.concat([user_df, new_df], ignore_index=True)

    res_df = df.copy()
    res_df[col_label] = pos_value
    return (
        res_df.groupby(col_user)
        .apply(sample_items)
        .reset_index(drop=True)
        .rename(columns={col_label: col_feedback})
    )


def has_columns(df, columns):
    """Check if DataFrame has necessary columns

    Args:
        df (pandas.DataFrame): DataFrame
        columns (list(str): columns to check for

    Returns:
        bool: True if DataFrame has specified columns.
    """

    result = True
    for column in columns:
        if column not in df.columns:
            logger.error("Missing column: {} in DataFrame".format(column))
            result = False

    return result


def has_same_base_dtype(df_1, df_2, columns=None):
    """Check if specified columns have the same base dtypes across both DataFrames

    Args:
        df_1 (pandas.DataFrame): first DataFrame
        df_2 (pandas.DataFrame): second DataFrame
        columns (list(str)): columns to check, None checks all columns

    Returns:
        bool: True if DataFrames columns have the same base dtypes.
    """

    if columns is None:
        if any(set(df_1.columns).symmetric_difference(set(df_2.columns))):
            logger.error(
                "Cannot test all columns because they are not all shared across DataFrames"
            )
            return False
        columns = df_1.columns

    if not (
        has_columns(df=df_1, columns=columns) and has_columns(df=df_2, columns=columns)
    ):
        return False

    result = True
    for column in columns:
        if df_1[column].dtype.type.__base__ != df_2[column].dtype.type.__base__:
            logger.error("Columns {} do not have the same base datatype".format(column))
            result = False

    return result


class PandasHash:
    """Wrapper class to allow pandas objects (DataFrames or Series) to be hashable"""

    # reserve space just for a single pandas object
    __slots__ = "pandas_object"

    def __init__(self, pandas_object):
        """Initialize class

        Args:
            pandas_object (pandas.DataFrame|pandas.Series): pandas object
        """

        if not isinstance(pandas_object, (pd.DataFrame, pd.Series)):
            raise TypeError("Can only wrap pandas DataFrame or Series objects")
        self.pandas_object = pandas_object

    def __eq__(self, other):
        """Overwrite equality comparison

        Args:
            other (pandas.DataFrame|pandas.Series): pandas object to compare

        Returns:
            bool: whether other object is the same as this one
        """

        return hash(self) == hash(other)

    def __hash__(self):
        """Overwrite hash operator for use with pandas objects

        Returns:
            int: hashed value of object
        """

        hashable = tuple(self.pandas_object.values.tobytes())
        if isinstance(self.pandas_object, pd.DataFrame):
            hashable += tuple(self.pandas_object.columns)
        else:
            hashable += tuple(self.pandas_object.name)
        return hash(hashable)


def lru_cache_df(maxsize, typed=False):
    """Least-recently-used cache decorator for pandas Dataframes.

    Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can
    save time when an expensive or I/O bound function is periodically called with the same arguments.

    Inspired in the `lru_cache function <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.

    Args:
        maxsize (int|None): max size of cache, if set to None cache is boundless
        typed (bool): arguments of different types are cached separately
    """

    def to_pandas_hash(val):
        """Return PandaHash object if input is a DataFrame otherwise return input unchanged"""
        return PandasHash(val) if isinstance(val, pd.DataFrame) else val

    def from_pandas_hash(val):
        """Extract DataFrame if input is PandaHash object otherwise return input unchanged"""
        return val.pandas_object if isinstance(val, PandasHash) else val

    def decorating_function(user_function):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            # convert DataFrames in args and kwargs to PandaHash objects
            args = tuple([to_pandas_hash(a) for a in args])
            kwargs = {k: to_pandas_hash(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            # get DataFrames from PandaHash objects in args and kwargs
            args = tuple([from_pandas_hash(a) for a in args])
            kwargs = {k: from_pandas_hash(v) for k, v in kwargs.items()}
            return user_function(*args, **kwargs)

        # retain lru_cache attributes
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorating_function
