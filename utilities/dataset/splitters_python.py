import numpy as np
import pandas as pd


def pandas_random_split(data, ratio=[0.8, 0.2], seed=1234, resample=True):
    splits = _split_pandas_data_with_ratios(data, ratio, seed=seed, resample=resample)
    return splits


def pandas_chrono_split(
    data,
    ratio=[0.8, 0.2],
    filter_by="user",
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_timestamp="Timestamp",
):
    if col_timestamp not in data.columns:
        raise ValueError("There is no column with temporal data")

    split_by_column = col_user if filter_by == "user" else col_item

    # Sort data by timestamp.
    data = data.sort_values(
        by=[split_by_column, col_timestamp], axis=0, ascending=False
    )

    # TODO: add min_rating
    # min_rating > 1:
    # data = min_rating_filter(data, min_rating=min_rating,
    #                             filter_by=filter_by,
    #                             col_user=col_user, col_item=col_item)

    num_of_splits = len(ratio)
    splits = [pd.DataFrame({})] * num_of_splits
    df_grouped = data.sort_values(col_timestamp).groupby(split_by_column)
    for name, group in df_grouped:
        group_splits = _split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, resample=False
        )
        for x in range(num_of_splits):
            splits[x] = pd.concat([splits[x], group_splits[x]])

    return splits


def _split_pandas_data_with_ratios(data, ratio, seed=1234, resample=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train
        -validation-and-test

    Args:
        data (pd.DataFrame): Pandas data frame to be split.
        ratio (list of floats): list of ratios for split.
        seed (int): random seed.
        resample (bool): whether data will be resampled when being split.

    Returns:
        List of data frames which are split by the given specifications.
    """
    split_index = np.cumsum(ratio).tolist()[:-1]

    if resample:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    return splits
