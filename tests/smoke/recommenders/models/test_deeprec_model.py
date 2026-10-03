# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split

try:
    from recommenders.models.deeprec.deeprec_utils import (
        download_deeprec_resources,
        prepare_hparams,
    )
    from recommenders.models.deeprec.models.base_model import BaseModel
    from recommenders.models.deeprec.models.pytorch.dkn import DKN
    from recommenders.models.deeprec.models.pytorch.dkn_item2item import DKNItem2Item
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
    from recommenders.datasets.amazon_reviews import (
        download_and_extract,
        data_preprocessing,
    )
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
    from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF

except ImportError:
    pass  # disable error while collecting tests for non-gpu environments

try:
    from recommenders.models.deeprec.models.sequential.sum import SUMModel
except ImportError:
    pass  # disable error while collecting tests for SUMModel


@pytest.mark.gpu
def test_model_dkn(deeprec_resource_path, tmp_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    train_file = os.path.join(data_path, "train_mind_demo.txt")
    valid_file = os.path.join(data_path, "valid_mind_demo.txt")
    output_file = os.path.join(tmp_path, "output.txt")

    download_deeprec_resources(
        "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    model = DKN(
        news_feature_file=os.path.join(data_path, "doc_feature.txt"),
        user_history_file=os.path.join(data_path, "user_history.txt"),
        word_embedding_file=os.path.join(data_path, "word_embeddings_100.npy"),
        entity_embedding_file=os.path.join(data_path, "TransE_entity2vec_100.npy"),
        context_embedding_file=os.path.join(data_path, "TransE_context2vec_100.npy"),
        seed=42,
    )

    assert model.fit(train_file, valid_file, epochs=1, learning_rate=0.0001) is model
    res = model.run_eval(valid_file)
    assert set(res) == {"auc", "group_auc", "mean_mrr", "ndcg@5", "ndcg@10"}
    assert all(0 <= value <= 1 for value in res.values())

    model.predict(valid_file, output_file)
    with open(valid_file) as rd:
        n_instances = sum(1 for _ in rd)
    with open(output_file) as rd:
        preds = [float(line) for line in rd]
    assert len(preds) == n_instances
    assert all(0 <= pred <= 1 for pred in preds)


@pytest.mark.gpu
def test_model_dkn_item2item(deeprec_resource_path, tmp_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    news_feature_file = os.path.join(data_path, "doc_feature.txt")
    # 20 news IDs: 4 groups of a source, its related target and 3 unrelated ones.
    doc_list_file = os.path.join(data_path, "doc_list.txt")
    embedding_file = os.path.join(tmp_path, "embedding.txt")

    download_deeprec_resources(
        "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    model = DKNItem2Item(
        news_feature_file=news_feature_file,
        word_embedding_file=os.path.join(data_path, "word_embeddings_100.npy"),
        neg_num=3,
        entity_embedding_file=os.path.join(data_path, "TransE_entity2vec_100.npy"),
        context_embedding_file=os.path.join(data_path, "TransE_context2vec_100.npy"),
        seed=42,
    )

    fitted = model.fit(
        doc_list_file, doc_list_file, epochs=2, batch_size=2, max_grad_norm=0.5
    )
    assert fitted is model
    res = model.run_eval(doc_list_file, batch_size=2)
    assert set(res) == {"group_auc", "mean_mrr", "ndcg@5", "ndcg@10"}
    assert all(0 <= value <= 1 for value in res.values())

    model.run_get_embedding(news_feature_file, embedding_file)
    with open(news_feature_file) as rd:
        n_news = sum(1 for _ in rd)
    with open(embedding_file) as rd:
        norms = [
            sum(float(v) ** 2 for v in line.split(" ")[1].split(",")) ** 0.5
            for line in rd
        ]
    assert len(norms) == n_news
    assert norms == pytest.approx([1.0] * n_news, abs=1e-4)


@pytest.mark.gpu
def test_model_sum(deeprec_resource_path, deeprec_config_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
    yaml_file = os.path.join(deeprec_config_path, "sum.yaml")
    train_file = os.path.join(data_path, r"train_data")
    valid_file = os.path.join(data_path, r"valid_data")
    test_file = os.path.join(data_path, r"test_data")
    output_file = os.path.join(data_path, "output.txt")
    train_num_ngs = (
        4  # number of negative instances with a positive instance for training
    )
    valid_num_ngs = (
        4  # number of negative instances with a positive instance for validation
    )
    test_num_ngs = (
        9  # number of negative instances with a positive instance for testing
    )

    if not os.path.exists(train_file):
        user_vocab = os.path.join(data_path, r"user_vocab.pkl")
        item_vocab = os.path.join(data_path, r"item_vocab.pkl")
        cate_vocab = os.path.join(data_path, r"category_vocab.pkl")
        reviews_name = "reviews_Movies_and_TV_5.json"
        meta_name = "meta_Movies_and_TV.json"
        reviews_file = os.path.join(data_path, reviews_name)
        meta_file = os.path.join(data_path, meta_name)
        sample_rate = (
            0.005  # sample a small item set for training and testing here for example
        )

        input_files = [
            reviews_file,
            meta_file,
            train_file,
            valid_file,
            test_file,
            user_vocab,
            item_vocab,
            cate_vocab,
        ]
        download_and_extract(reviews_name, reviews_file)
        download_and_extract(meta_name, meta_file)
        data_preprocessing(
            *input_files,
            sample_rate=sample_rate,
            valid_num_ngs=valid_num_ngs,
            test_num_ngs=test_num_ngs
        )

    hparams = prepare_hparams(
        yaml_file, learning_rate=0.01, epochs=1, train_num_ngs=train_num_ngs
    )
    assert hparams is not None

    input_creator = SequentialIterator
    model = SUMModel(hparams, input_creator)
    assert model.run_eval(valid_file, num_ngs=valid_num_ngs) is not None
    assert isinstance(
        model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs), BaseModel
    )
    assert model.predict(valid_file, output_file) is not None


@pytest.mark.gpu
def test_model_lightgcn(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    user_file = os.path.join(data_path, r"user_embeddings.csv")
    item_file = os.path.join(data_path, r"item_embeddings.csv")

    df = movielens.load_pandas_df(size="100k")
    train, test = python_stratified_split(df, ratio=0.75)

    data = ImplicitCF(train=train, test=test)

    model = LightGCN(
        n_users=data.n_users,
        n_items=data.n_items,
        norm_adj=data.get_norm_adj_mat(),
    )
    model.fit(data, epochs=1)
    assert model.run_eval() is not None
    assert model.recommend_k_items(test) is not None
    model.infer_embedding(user_file, item_file)
    assert os.path.getsize(user_file) != 0
    assert os.path.getsize(item_file) != 0
