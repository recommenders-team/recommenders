import numpy as np
import pandas as pd


def pandas_random_split(self, data, **kwargs):
    if self.multi_split:
        splits = _split_pandas_data_with_ratios(
            data, self.ratio, resample=True, seed=self.seed
        )
        return splits
    else:
        return sk_split(
            data, test_size=None, train_size=self.ratio, random_state=self.seed
        )


def pandas_chrono_split(self, data, **kwargs):
    split_by_column = self.col_user if self.filter_by == "user" else self.col_item

    # Sort data by timestamp.
    data = data.sort_values(
        by=[split_by_column, self.col_timestamp], axis=0, ascending=False
    )

    ratio = self.ratio if self.multi_split else [self.ratio, 1 - self.ratio]

    if self.min_rating > 1:
        data = min_rating_filter(
            data,
            min_rating=self.min_rating,
            filter_by=self.filter_by,
            col_user=self.col_user,
            col_item=self.col_item,
        )