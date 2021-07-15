# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import numpy as np
import pytest

try:
    from reco_utils.recommender.ncf.ncf_singlenode import NCF
    from reco_utils.recommender.ncf.dataset import Dataset
    from reco_utils.utils.constants import (
        DEFAULT_USER_COL,
        DEFAULT_ITEM_COL,
        SEED,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


N_NEG = 5
N_NEG_TEST = 10


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_type, n_users, n_items", [("NeuMF", 1, 1), ("GMF", 10, 10), ("MLP", 4, 8)]
)
def test_init(model_type, n_users, n_items):
    model = NCF(
        n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED
    )
    # model type
    assert model.model_type == model_type.lower()
    # number of users in dataset
    assert model.n_users == n_users
    # number of items in dataset
    assert model.n_items == n_items
    # dimension of gmf user embedding
    assert model.embedding_gmf_P.shape == [n_users, model.n_factors]
    # dimension of gmf item embedding
    assert model.embedding_gmf_Q.shape == [n_items, model.n_factors]
    # dimension of mlp user embedding
    assert model.embedding_mlp_P.shape == [n_users, model.n_factors]
    # dimension of mlp item embedding
    assert model.embedding_mlp_Q.shape == [n_items, model.n_factors]

    # TODO: more parameters


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_type, n_users, n_items", [("NeuMF", 5, 5), ("GMF", 5, 5), ("MLP", 5, 5)]
)
def test_regular_save_load(model_type, n_users, n_items):
    ckpt = ".%s" % model_type
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)

    model = NCF(
        n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED
    )
    model.save(ckpt)
    if model.model_type == "neumf":
        P = model.sess.run(model.embedding_gmf_P)
        Q = model.sess.run(model.embedding_mlp_Q)
    elif model.model_type == "gmf":
        P = model.sess.run(model.embedding_gmf_P)
        Q = model.sess.run(model.embedding_gmf_Q)
    elif model.model_type == "mlp":
        P = model.sess.run(model.embedding_mlp_P)
        Q = model.sess.run(model.embedding_mlp_Q)

    del model
    model = NCF(
        n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1, seed=SEED
    )

    if model.model_type == "neumf":
        model.load(neumf_dir=ckpt)
        P_ = model.sess.run(model.embedding_gmf_P)
        Q_ = model.sess.run(model.embedding_mlp_Q)
    elif model.model_type == "gmf":
        model.load(gmf_dir=ckpt)
        P_ = model.sess.run(model.embedding_gmf_P)
        Q_ = model.sess.run(model.embedding_gmf_Q)
    elif model.model_type == "mlp":
        model.load(mlp_dir=ckpt)
        P_ = model.sess.run(model.embedding_mlp_P)
        Q_ = model.sess.run(model.embedding_mlp_Q)

    # test load function
    assert np.array_equal(P, P_)
    assert np.array_equal(Q, Q_)

    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)


@pytest.mark.gpu
@pytest.mark.parametrize("n_users, n_items", [(5, 5), (4, 8)])
def test_neumf_save_load(n_users, n_items):
    model_type = "gmf"
    ckpt_gmf = ".%s" % model_type
    if os.path.exists(ckpt_gmf):
        shutil.rmtree(ckpt_gmf)
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.save(ckpt_gmf)
    P_gmf = model.sess.run(model.embedding_gmf_P)
    Q_gmf = model.sess.run(model.embedding_gmf_Q)
    del model

    model_type = "mlp"
    ckpt_mlp = ".%s" % model_type
    if os.path.exists(ckpt_mlp):
        shutil.rmtree(ckpt_mlp)
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.save(".%s" % model_type)
    P_mlp = model.sess.run(model.embedding_mlp_P)
    Q_mlp = model.sess.run(model.embedding_mlp_Q)
    del model

    model_type = "neumf"
    model = NCF(n_users=n_users, n_items=n_items, model_type=model_type, n_epochs=1)
    model.load(gmf_dir=ckpt_gmf, mlp_dir=ckpt_mlp)

    P_gmf_ = model.sess.run(model.embedding_gmf_P)
    Q_gmf_ = model.sess.run(model.embedding_gmf_Q)

    P_mlp_ = model.sess.run(model.embedding_mlp_P)
    Q_mlp_ = model.sess.run(model.embedding_mlp_Q)

    assert np.array_equal(P_gmf, P_gmf_)
    assert np.array_equal(Q_gmf, Q_gmf_)
    assert np.array_equal(P_mlp, P_mlp_)
    assert np.array_equal(Q_mlp, Q_mlp_)

    if os.path.exists(ckpt_gmf):
        shutil.rmtree(ckpt_gmf)
    if os.path.exists(ckpt_mlp):
        shutil.rmtree(ckpt_mlp)

    # TODO: test loading fc-concat


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", ["NeuMF", "GMF", "MLP"])
def test_fit(python_dataset_ncf, model_type):
    train, test = python_dataset_ncf
    data = Dataset(train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST)
    model = NCF(
        n_users=data.n_users, n_items=data.n_items, model_type=model_type, n_epochs=1
    )
    model.fit(data)


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", ["NeuMF", "GMF", "MLP"])
def test_predict(python_dataset_ncf, model_type):
    # test data format
    train, test = python_dataset_ncf
    data = Dataset(train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST)
    model = NCF(
        n_users=data.n_users, n_items=data.n_items, model_type=model_type, n_epochs=1
    )
    model.fit(data)

    test_users, test_items = list(test[DEFAULT_USER_COL]), list(test[DEFAULT_ITEM_COL])

    assert type(model.predict(test_users[0], test_items[0])) == float

    res = model.predict(test_users, test_items, is_list=True)

    assert type(res) == list
    assert len(res) == len(test)
