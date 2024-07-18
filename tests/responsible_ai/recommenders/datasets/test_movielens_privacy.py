# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


from recommenders.datasets import movielens


def test_movielens_privacy():
    """Check that there are no privacy concerns. In Movielens, we check that all the
    userID are numbers.
    """
    df = movielens.load_pandas_df(size="100k")
    users = df["userID"].values.tolist()

    assert all(isinstance(x, int) for x in users) is True
