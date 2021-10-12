# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os

from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.ssept import SSEPT
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet


@pytest.mark.gpu
def test_prepare_data():
    data_dir = "/recsys_data/RecSys/SASRec-tf2/data/"
    dataset = "ae"
    inp_file = os.path.join(data_dir, dataset + ".txt")

    # initiate a dataset class
    data = SASRecDataSet(filename=inp_file)

    # create train, validation and test splits
    data.split()

    assert len(data.user_train) > 0
    assert len(data.user_valid) > 0
    assert len(data.user_test) > 0


@pytest.mark.gpu
def test_sampler():
    data_dir = "/recsys_data/RecSys/SASRec-tf2/data/"
    dataset = "ae"
    batch_size = 8
    maxlen = 50
    inp_file = os.path.join(data_dir, dataset + ".txt")

    # initiate a dataset class
    data = SASRecDataSet(filename=inp_file)

    # create train, validation and test splits
    data.split()

    sampler = WarpSampler(
        data.user_train,
        data.usernum,
        data.itemnum,
        batch_size=batch_size,
        maxlen=maxlen,
        n_workers=3,
    )
    u, seq, pos, neg = sampler.next_batch()

    assert len(u) == batch_size
    assert len(seq) == batch_size
    assert len(pos) == batch_size
    assert len(neg) == batch_size


@pytest.mark.gpu
def test_sasrec():
    # Amazon Electronics Data
    itemnum = 85930
    maxlen = 50
    num_blocks = 2
    hidden_units = 100
    num_heads = 1
    dropout_rate = 0.1
    l2_emb = 0.0
    num_neg_test = 100

    model = SASREC(
        item_num=itemnum,
        seq_max_len=maxlen,
        num_blocks=num_blocks,
        embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=num_heads,
        dropout_rate=dropout_rate,
        conv_dims=[100, 100],
        l2_reg=l2_emb,
        num_neg_test=num_neg_test,
    )

    assert model.encoder is not None
    assert model.item_embedding_layer is not None


@pytest.mark.gpu
def test_ssept():
    # Amazon Electronics Data
    itemnum = 85930
    usernum = 63114
    maxlen = 50
    num_blocks = 2
    hidden_units = 100
    num_heads = 1
    dropout_rate = 0.1
    l2_emb = 0.0
    num_neg_test = 100

    model = SSEPT(
        item_num=itemnum,
        user_num=usernum,
        seq_max_len=maxlen,
        num_blocks=num_blocks,
        # embedding_dim=hidden_units,  # optional
        user_embedding_dim=hidden_units,
        item_embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=num_heads,
        dropout_rate=dropout_rate,
        conv_dims=[200, 200],
        l2_reg=l2_emb,
        num_neg_test=num_neg_test,
    )

    assert model.encoder is not None
    assert model.item_embedding_layer is not None
    assert model.user_embedding_layer is not None
