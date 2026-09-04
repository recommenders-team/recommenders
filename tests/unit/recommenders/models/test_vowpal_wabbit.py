# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest
from unittest import mock
import numpy as np
import pandas as pd

from recommenders.models.vowpal_wabbit.vw import VW

try:
    import vowpalwabbit
except ImportError:
    vowpalwabbit = None

requires_vw = pytest.mark.skipif(
    vowpalwabbit is None, reason="vowpalwabbit not installed"
)


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


@pytest.mark.experimental
def test_vw_init_del():
    model = VW()
    tempdir = model.tempdir.name
    assert os.path.exists(tempdir)

    del model
    assert not os.path.exists(tempdir)


@pytest.mark.experimental
def test_to_vw_cmd():
    expected = "-l 0.1 --l1 0.2 --loss_function logistic --holdout_off --rank 3 -t"
    params = dict(
        l=0.1,
        l1=0.2,
        loss_function="logistic",
        holdout_off=True,
        quiet=False,
        rank=3,
        t=True,
    )
    assert VW.to_vw_params(params=params) == expected


@pytest.mark.experimental
def test_parse_train_cmd(model):
    expected = (
        f"--loss_function logistic --oaa 5 -f {model.model_file} -d {model.train_file}"
    )
    params = dict(loss_function="logistic", oaa=5, f="test", d="data", quiet=False)
    assert model.parse_train_params(params=params) == expected


@pytest.mark.experimental
def test_parse_test_cmd(model):
    expected = (
        f"--loss_function logistic -d {model.test_file} --quiet "
        f"-i {model.model_file} -p {model.prediction_file} -t"
    )
    params = dict(
        loss_function="logistic", i="test", oaa=5, d="data", test_only=True, quiet=True
    )
    assert model.parse_test_params(params=params) == expected


@pytest.mark.experimental
@pytest.mark.parametrize(
    "train, expected",
    [
        (True, ["1 0|user 1 |item 8", "5 1|user 3 |item 7", "3 2|user 2 |item 7"]),
        (False, [" 0|user 1 |item 8", " 1|user 3 |item 7", " 2|user 2 |item 7"]),
    ],
)
def test_to_vw_file(model, df, train, expected):
    model.to_vw_file(df, train=train)
    path = model.train_file if train else model.test_file
    with open(path, "r") as f:
        assert f.read().splitlines() == expected


@pytest.mark.experimental
def test_to_vw_file_logistic(df):
    model = VW(col_user="user", col_item="item", loss_function="logistic")
    model.to_vw_file(df, train=True)
    with open(model.train_file, "r") as f:
        labels = [line.split(" ")[0] for line in f.read().splitlines()]
    assert labels == ["-1", "1", "1"]


@pytest.mark.experimental
def test_fit_and_predict(model, df):
    # generate fake predictions
    with open(model.prediction_file, "w") as f:
        f.writelines(["1 0\n", "3 1\n", "5 2\n"])

    # patch the vw bindings so no model is actually trained
    with mock.patch("recommenders.models.vowpal_wabbit.vw.vowpalwabbit"):
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


@pytest.mark.experimental
@requires_vw
def test_fit_and_predict_with_vw(model, df):
    model.fit(df)
    result = model.predict(df)

    assert list(result.columns) == ["user", "item", "rating", "timestamp", "prediction"]
    assert len(result) == len(df)
    assert np.isfinite(result["prediction"]).all()


@pytest.mark.experimental
@requires_vw
def test_recommend_k_items(model, df):
    model.fit(df)
    top_k = model.recommend_k_items(df, top_k=1, remove_seen=True)

    # items seen in training are 7 and 8; each user has rated exactly one of them
    assert list(top_k.columns) == ["user", "item", "prediction"]
    assert set(zip(top_k["user"], top_k["item"])) == {(1, 7), (3, 8), (2, 8)}


@pytest.mark.experimental
@requires_vw
def test_logistic_predictions(df):
    model = VW(
        col_user="user", col_item="item", loss_function="logistic", link="logistic"
    )
    model.fit(df)
    result = model.predict(df)

    assert result["prediction"].between(0, 1).all()


@pytest.mark.experimental
@requires_vw
def test_multiple_passes(df):
    # passes needs a cache during training and must not be used at prediction time
    model = VW(col_user="user", col_item="item", passes=3, c=True)
    model.fit(df)
    result = model.predict(df)

    assert "--passes" not in model.test_params
    assert len(result) == len(df)


@pytest.mark.experimental
def test_n_jobs_is_not_a_vw_option():
    model = VW(n_jobs=4, q="ui")
    assert "n_jobs" not in model.train_params
    assert "n_jobs" not in model.test_params


@pytest.mark.experimental
@requires_vw
@pytest.mark.parametrize("n_jobs", [2, 3])
def test_predict_in_parallel(n_jobs):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        dict(
            user=rng.integers(1, 20, 200),
            item=rng.integers(1, 30, 200),
            rating=rng.integers(1, 6, 200),
        )
    )
    single = VW(col_user="user", col_item="item", q="ui")
    parallel = VW(col_user="user", col_item="item", q="ui", n_jobs=n_jobs)
    single.fit(data)
    parallel.fit(data)

    expected = single.recommend_k_items(data, top_k=5, remove_seen=True)
    result = parallel.recommend_k_items(data, top_k=5, remove_seen=True)

    pd.testing.assert_frame_equal(result, expected)
