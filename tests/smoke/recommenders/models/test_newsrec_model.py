# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest

try:
    from recommenders.models.newsrec.models.lstur import LSTURModel
    from recommenders.models.newsrec.models.naml import NAMLModel
    from recommenders.models.newsrec.models.npa import NPAModel
    from recommenders.models.newsrec.models.nrms import NRMSModel
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


def check_model(model, mind_demo):
    assert model.run_eval(*mind_demo["valid"]) is not None
    assert (
        model.fit(*mind_demo["train"], *mind_demo["valid"], epochs=1, step_limit=10)
        is model
    )


@pytest.mark.gpu
def test_model_nrms(mind_demo):
    model = NRMSModel(mind_demo["embedding"], mind_demo["word_dict"], seed=42)
    check_model(model, mind_demo)


@pytest.mark.gpu
def test_model_naml(mind_demo):
    model = NAMLModel(
        mind_demo["embedding_all"],
        mind_demo["word_dict_all"],
        mind_demo["vert_dict"],
        mind_demo["subvert_dict"],
        seed=42,
    )
    check_model(model, mind_demo)


@pytest.mark.gpu
def test_model_lstur(mind_demo):
    model = LSTURModel(
        mind_demo["embedding"], mind_demo["word_dict"], mind_demo["uid2index"], seed=42
    )
    check_model(model, mind_demo)


@pytest.mark.gpu
def test_model_npa(mind_demo):
    model = NPAModel(
        mind_demo["embedding"], mind_demo["word_dict"], mind_demo["uid2index"], seed=42
    )
    check_model(model, mind_demo)
