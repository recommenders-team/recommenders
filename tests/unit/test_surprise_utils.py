# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Test utils for Surprise algos
"""
import numpy as np
import pytest

from reco_utils.recommender.surprise.surprise_utils import (
    compute_predictions,
    compute_all_predictions
)
from surprise.prediction_algorithms import SVD
from tests.unit.test_python_evaluation import python_data

TOL = 0.0001


def test_compute_predictions(python_data):
    rating_true, _ = python_data
    svd = SVD()
    svd.fit(rating_true)

    preds = compute_predictions(svd, rating_true)
    assert set(preds.columns) == {'userID', 'itemID', 'prediction'}
    user = rating_true.loc[0, 'userID']
    item = rating_true.loc[0, 'itemID']
    assert preds[(preds['userID'] == user) & (preds['itemID'] == item)]['prediction'] == \
           pytest.approx(svd.predict(user, item), rel=TOL)

    preds = compute_predictions(svd, rating_true.rename(columns={'userID': 'uid', 'itemID': 'iid'}),
                        usercol='uid', itemcol='iid')
    assert set(preds.columns) == {'uid', 'iid', 'prediction'}
    user = rating_true.loc[1, 'userID']
    item = rating_true.loc[1, 'itemID']
    assert preds[(preds['uid'] == user) & (preds['iid'] == item)]['prediction'] == \
           pytest.approx(svd.predict(user, item), rel=TOL)

