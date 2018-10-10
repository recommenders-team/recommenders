import os
import sys
import numpy as np
import pandas as pd
import pytest

# TODO: better solution??
root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
sys.path.append(root)
from utilities.dataset.splitters_python import pandas_random_split, pandas_chrono_split


def test_pandas_random_split(load_pandas_dummy_dataset):
    df = load_pandas_dummy_dataset
    split1, split2 = pandas_random_split(df, ratios=[0.5, 0.5])
    assert split1.shape[0] == 5
    assert split2.shape[1] == 5


def test_pandas_chrono_split(load_pandas_dummy_timestamp_dataset, header):
    df = load_pandas_dummy_timestamp_dataset
    split1, split2 = pandas_chrono_split(df, ratios=[0.8, 0.2])
    assert split1.shape[0] == 8
    assert split2.shape[1] == 2

    # Make sure that it splits chronological
    time1 = split1[header["col_timestamp"]].values
    time2 = split2[header["col_timestamp"]].values
    assert all(time1) < all(time2)

