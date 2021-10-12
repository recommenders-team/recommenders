# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)

try:
    from recommenders.utils.tf_utils import pandas_input_fn, MODEL_DIR
    from recommenders.models.wide_deep.wide_deep_utils import (
        build_model,
        build_feature_columns,
    )
    import tensorflow as tf
except ImportError:
    pass  # skip this import if we are in cpu environment


ITEM_FEAT_COL = "itemFeat"


@pytest.fixture(scope="module")
def pd_df():
    df = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5],
            ITEM_FEAT_COL: [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [1, 1, 1],
                [4, 4, 4],
                [5, 5, 5],
            ],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    users = df.drop_duplicates(DEFAULT_USER_COL)[DEFAULT_USER_COL].values
    items = df.drop_duplicates(DEFAULT_ITEM_COL)[DEFAULT_ITEM_COL].values
    return df, users, items


@pytest.mark.gpu
def test_wide_model(pd_df, tmp):
    data, users, items = pd_df

    # Test wide model
    # Test if wide column has two original features and one crossed feature
    wide_columns, _ = build_feature_columns(
        users, items, model_type="wide", crossed_feat_dim=10
    )
    assert len(wide_columns) == 3
    # Check crossed feature dimension
    assert wide_columns[2].hash_bucket_size == 10
    # Check model type
    model = build_model(
        os.path.join(tmp, "wide_" + MODEL_DIR), wide_columns=wide_columns
    )
    assert isinstance(model, tf.estimator.LinearRegressor)
    # Test if model train works
    model.train(
        input_fn=pandas_input_fn(
            df=data,
            y_col=DEFAULT_RATING_COL,
            batch_size=1,
            num_epochs=None,
            shuffle=True,
        ),
        steps=1,
    )

    # Close the event file so that the model folder can be cleaned up.
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()


@pytest.mark.gpu
def test_deep_model(pd_df, tmp):
    data, users, items = pd_df

    # Test if deep columns have user and item features
    _, deep_columns = build_feature_columns(users, items, model_type="deep")
    assert len(deep_columns) == 2
    # Check model type
    model = build_model(
        os.path.join(tmp, "deep_" + MODEL_DIR), deep_columns=deep_columns
    )
    assert isinstance(model, tf.estimator.DNNRegressor)
    # Test if model train works
    model.train(
        input_fn=pandas_input_fn(
            df=data, y_col=DEFAULT_RATING_COL, batch_size=1, num_epochs=1, shuffle=False
        )
    )

    # Close the event file so that the model folder can be cleaned up.
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()


@pytest.mark.gpu
def test_wide_deep_model(pd_df, tmp):
    data, users, items = pd_df

    # Test if wide and deep columns have correct features
    wide_columns, deep_columns = build_feature_columns(
        users, items, model_type="wide_deep"
    )
    assert len(wide_columns) == 3
    assert len(deep_columns) == 2
    # Check model type
    model = build_model(
        os.path.join(tmp, "wide_deep_" + MODEL_DIR),
        wide_columns=wide_columns,
        deep_columns=deep_columns,
    )
    assert isinstance(model, tf.estimator.DNNLinearCombinedRegressor)
    # Test if model train works
    model.train(
        input_fn=pandas_input_fn(
            df=data,
            y_col=DEFAULT_RATING_COL,
            batch_size=1,
            num_epochs=None,
            shuffle=True,
        ),
        steps=1,
    )

    # Close the event file so that the model folder can be cleaned up.
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()
