# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymanopt")

from recommenders.models.rlrmc.RLRMCalgorithm import RLRMCalgorithm
from recommenders.models.rlrmc.RLRMCdataset import RLRMCdataset

pytestmark = pytest.mark.experimental


def _factor_model(mean):
    model = RLRMCalgorithm(
        rank=3,
        C=0.01,
        model_param={"num_row": 4, "num_col": 5, "train_mean": mean},
    )
    model.user2id = {f"user-{i}": i for i in range(4)}
    model.item2id = {f"item-{i}": i for i in range(5)}
    # Noncontiguous factors also exercise the compiled sparse scoring kernel.
    model.L = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2] / 4
    model.R = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2] / 8 - 1
    return model


@pytest.mark.parametrize("mean", [0.0, 3.25, -2.5])
@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize(
    "pairs",
    [
        [(3, 4), (0, 2), (3, 1), (1, 0)],
        [(3, 4), (0, 2), (3, 4), (1, 0), (0, 2), (3, 1)],
        [(2, 3)] * 5,
    ],
    ids=["unsorted-unique", "interleaved-repeats", "one-repeated-pair"],
)
def test_rlrmc_predict_pair_scores(mean, low_memory, pairs):
    model = _factor_model(mean)
    users = [f"user-{u}" for u, _ in pairs]
    items = [f"item-{i}" for _, i in pairs]
    expected = [
        mean + sum(float(x) * float(y) for x, y in zip(model.L[u], model.R[i]))
        for u, i in pairs
    ]
    actual = model.predict(users, items, low_memory=low_memory)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.shape == (len(pairs),)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("low_memory", [False, True])
def test_rlrmc_predict_independent_of_batching(low_memory):
    model = _factor_model(3.25)
    users = ["user-3", "user-1", "user-3", "user-0", "user-1"]
    items = ["item-4", "item-2", "item-4", "item-0", "item-2"]
    expected = np.concatenate(
        [model.predict([u], [i], low_memory=low_memory) for u, i in zip(users, items)]
    )
    actual = model.predict(users, items, low_memory=low_memory)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("low_memory", [False, True])
def test_rlrmc_predict_empty(low_memory):
    actual = _factor_model(3.25).predict([], [], low_memory=low_memory)
    assert actual.shape == (0,)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("low_memory", [False, True])
def test_rlrmc_predict_mismatched_pairs(low_memory, capsys):
    model = _factor_model(3.25)
    assert model.predict(["user-0"], [], low_memory=low_memory) is None
    assert "Dimension mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("mean_center", [False, True])
@pytest.mark.parametrize("low_memory", [False, True])
def test_rlrmc_predict_after_native_fit(mean_center, low_memory):
    train = pd.DataFrame(
        [(u, i, 3.0 + 0.2 * u - 0.1 * i) for u in range(4) for i in range(3)],
        columns=["userID", "itemID", "rating"],
    )
    data = RLRMCdataset(train, mean_center=mean_center)
    model = RLRMCalgorithm(rank=1, C=0.01, model_param=data.model_param, maxiter=3)
    model.fit(data)
    users, items = [3, 0, 3, 1, 0], [2, 1, 2, 0, 1]
    expected = np.array(
        [
            sum(
                float(x) * float(y)
                for x, y in zip(model.L[data.user2id[u]], model.R[data.item2id[i]])
            )
            + data.model_param["train_mean"]
            for u, i in zip(users, items)
        ]
    )
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(
        model.predict(users, items, low_memory=low_memory), expected, rtol=1e-13
    )
