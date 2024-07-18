# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pandas as pd

from recommenders.datasets import criteo


def test_criteo_privacy(criteo_first_row):
    """Check that there are no privacy concerns. In Criteo, we check that the
    data is anonymized.
    """
    df = criteo.load_pandas_df(size="sample")
    assert df.loc[0].equals(pd.Series(criteo_first_row)) is True
