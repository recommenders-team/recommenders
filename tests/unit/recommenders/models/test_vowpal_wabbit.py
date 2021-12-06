# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from unittest import mock
import pandas as pd

from recommenders.models.vowpal_wabbit.vw import VW


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        dict(user=[1, 3, 2], item=[8, 7, 7], rating=[1, 5, 3], timestamp=[1, 2, 3])
    )


@pytest.fixture(scope="function")
def model():
    model = VW(col_user="user", col_item="item", col_prediction="prediction", q="ui")
    yield model
    del model


@pytest.mark.vw
def test_vw_init_del():
    model = VW()
    tempdir = model.tempdir.name
    assert os.path.exists(tempdir)

    del model
    assert not os.path.exists(tempdir)


@pytest.mark.vw
def test_to_vw_cmd():
    expected = [
        "vw",
        "-l",
        "0.1",
        "--l1",
        "0.2",
        "--loss_function",
        "logistic",
        "--holdout_off",
        "--rank",
        "3",
        "-t",
    ]
    params = dict(
        l=0.1,
        l1=0.2,
        loss_function="logistic",
        holdout_off=True,
        quiet=False,
        rank=3,
        t=True,
    )
    assert VW.to_vw_cmd(params=params) == expected


@pytest.mark.vw
def test_parse_train_cmd(model):
    expected = [
        "vw",
        "--loss_function",
        "logistic",
        "--oaa",
        "5",
        "-f",
        model.model_file,
        "-d",
        model.train_file,
    ]
    params = dict(loss_function="logistic", oaa=5, f="test", d="data", quiet=False)
    assert model.parse_train_params(params=params) == expected


@pytest.mark.vw
def test_parse_test_cmd(model):
    expected = [
        "vw",
        "--loss_function",
        "logistic",
        "-d",
        model.test_file,
        "--quiet",
        "-i",
        model.model_file,
        "-p",
        model.prediction_file,
        "-t",
    ]
    params = dict(
        loss_function="logistic", i="test", oaa=5, d="data", test_only=True, quiet=True
    )
    assert model.parse_test_params(params=params) == expected


@pytest.mark.vw
def test_to_vw_file(model, df):
    expected = ["1 0|user 1 |item 8", "5 1|user 3 |item 7", "3 2|user 2 |item 7"]
    model.to_vw_file(df, train=True)
    with open(model.train_file, "r") as f:
        assert f.read().splitlines() == expected
    del model


@pytest.mark.vw
def test_fit_and_predict(model, df):
    # generate fake predictions
    with open(model.prediction_file, "w") as f:
        f.writelines(["1 0\n", "3 1\n", "5 2\n"])

    # patch subprocess call to vw
    with mock.patch(
        "recommenders.models.vowpal_wabbit.vw.run"
    ) as mock_run:  # noqa: F841
        model.fit(df)
        result = model.predict(df)

    expected = dict(
        user=dict(enumerate([1, 3, 2])),
        item=dict(enumerate([8, 7, 7])),
        rating=dict(enumerate([1, 5, 3])),
        timestamp=dict(enumerate([1, 2, 3])),
        prediction=dict(enumerate([1, 3, 5])),
    )

    assert result.to_dict() == expected
