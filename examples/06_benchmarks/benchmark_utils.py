# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import cornac

try:
    from pyspark.ml.recommendation import ALS
    from pyspark.sql.types import StructType, StructField
    from pyspark.sql.types import FloatType, IntegerType, LongType
except ImportError:
    pass  # skip this import if we are not in a Spark environment
try:
    import surprise  # Put SVD surprise back in core deps when #2224 is fixed
except:
    pass

from recommenders.utils.timer import Timer
from recommenders.utils.constants import (
    COL_DICT,
    DEFAULT_K,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_TIMESTAMP_COL,
    SEED,
)
from recommenders.models.sar import SAR
from recommenders.models.cornac.bpr import BPR
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.models.surprise.surprise_utils import (
    predict,
    compute_ranking_predictions,
)
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var

try:
    from recommenders.utils.spark_utils import start_or_get_spark
    from recommenders.evaluation.spark_evaluation import (
        SparkRatingEvaluation,
        SparkRankingEvaluation,
    )
except (ImportError, NameError):
    pass  # skip this import if we are not in a Spark environment

try:
    from recommenders.models.deeprec.deeprec_utils import prepare_hparams
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
    from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from recommenders.models.ncf.ncf_singlenode import NCF
    from recommenders.models.ncf.dataset import Dataset as NCFDataset
    from recommenders.models.embdotbias.model import EmbeddingDotBias
    from recommenders.models.embdotbias.data_loader import RecoDataLoader
    from recommenders.models.embdotbias.training_utils import Trainer
    from recommenders.models.embdotbias.utils import cartesian_product, score

except ImportError:
    pass  # skip this import if we are not in a GPU environment

# Helpers
tmp_dir = TemporaryDirectory()
TRAIN_FILE = os.path.join(tmp_dir.name, "df_train.csv")
TEST_FILE = os.path.join(tmp_dir.name, "df_test.csv")


def prepare_training_als(train, test):
    schema = StructType(
        (
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
        )
    )
    spark = start_or_get_spark()
    return spark.createDataFrame(train, schema).cache()


def train_als(params, data):
    symbol = ALS(**params)
    with Timer() as t:
        model = symbol.fit(data)
    return model, t


def prepare_metrics_als(train, test):
    schema = StructType(
        (
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
        )
    )
    spark = start_or_get_spark()
    return (
        spark.createDataFrame(train, schema).cache(),
        spark.createDataFrame(test, schema).cache(),
    )


def predict_als(model, test):
    with Timer() as t:
        preds = model.transform(test)
    return preds, t


def recommend_k_als(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        # Get the cross join of all user-item pairs and score them.
        users = train.select(DEFAULT_USER_COL).distinct()
        items = train.select(DEFAULT_ITEM_COL).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)

        # Remove seen items
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            train.alias("train"),
            (dfs_pred[DEFAULT_USER_COL] == train[DEFAULT_USER_COL])
            & (dfs_pred[DEFAULT_ITEM_COL] == train[DEFAULT_ITEM_COL]),
            how="outer",
        )
        topk_scores = dfs_pred_exclude_train.filter(
            dfs_pred_exclude_train["train." + DEFAULT_RATING_COL].isNull()
        ).select(
            "pred." + DEFAULT_USER_COL,
            "pred." + DEFAULT_ITEM_COL,
            "pred." + DEFAULT_PREDICTION_COL,
        )
    return topk_scores, t


def prepare_training_svd(train, test):
    reader = surprise.Reader("ml-100k", rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1), reader=reader
    ).build_full_trainset()


def train_svd(params, data):
    model = surprise.SVD(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def predict_svd(model, test):
    with Timer() as t:
        preds = predict(
            model,
            test,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
        )
    return preds, t


def recommend_k_svd(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = compute_ranking_predictions(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def prepare_training_embdotbias(train, test):
    train_df = train.copy()
    train_df[DEFAULT_USER_COL] = train_df[DEFAULT_USER_COL].astype("str")
    train_df[DEFAULT_ITEM_COL] = train_df[DEFAULT_ITEM_COL].astype("str")
    data = RecoDataLoader.from_df(
        train_df,
        user_name=DEFAULT_USER_COL,
        item_name=DEFAULT_ITEM_COL,
        rating_name=DEFAULT_RATING_COL,
        valid_pct=0.1,
    )
    return data


def train_embdotbias(params, data):
    model = EmbeddingDotBias.from_classes(
        n_factors=params["n_factors"],
        classes=data.classes,
        user=DEFAULT_USER_COL,
        item=DEFAULT_ITEM_COL,
        y_range=params.get("y_range", [0, 5.5]),
    )

    with Timer() as t:
        trainer = Trainer(model=model)
        trainer.fit(data.train, data.valid, params["epochs"])
    return model, t


def prepare_metrics_embdotbias(train, test):
    train_df = train.copy()
    train_df[DEFAULT_USER_COL] = train_df[DEFAULT_USER_COL].astype("str")
    train_df[DEFAULT_ITEM_COL] = train_df[DEFAULT_ITEM_COL].astype("str")
    test_df = test.copy()
    test_df[DEFAULT_USER_COL] = test_df[DEFAULT_USER_COL].astype("str")
    test_df[DEFAULT_ITEM_COL] = test_df[DEFAULT_ITEM_COL].astype("str")
    return train_df, test_df


def predict_embdotbias(model, test):
    with Timer() as t:
        preds = score(
            model,
            test_df=test,
            user_col=DEFAULT_USER_COL,
            item_col=DEFAULT_ITEM_COL,
            prediction_col=DEFAULT_PREDICTION_COL,
        )
    return preds, t


def recommend_k_embdotbias(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    # Get all users/items known to the model
    total_users = model.classes[DEFAULT_USER_COL][1:]
    total_items = model.classes[DEFAULT_ITEM_COL][1:]
    test_users = test[DEFAULT_USER_COL].unique()
    test_users = np.intersect1d(test_users, total_users)
    users_items = cartesian_product(np.array(test_users), np.array(total_items))
    users_items = pd.DataFrame(
        users_items, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
    )
    if remove_seen:
        # Remove seen items
        training_removed = pd.merge(
            users_items,
            train.astype(str),
            on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
            how="left",
        )
        candidates = training_removed[training_removed[DEFAULT_RATING_COL].isna()][
            [DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        ]
    else:
        candidates = users_items
    with Timer() as t:
        topk_scores = score(
            model,
            test_df=candidates,
            user_col=DEFAULT_USER_COL,
            item_col=DEFAULT_ITEM_COL,
            prediction_col=DEFAULT_PREDICTION_COL,
            top_k=top_k,
        )
    return topk_scores, t


def prepare_training_ncf(df_train, df_test):
    train = df_train.sort_values([DEFAULT_USER_COL], axis=0, ascending=[True])
    test = df_test.sort_values([DEFAULT_USER_COL], axis=0, ascending=[True])
    test = test[df_test[DEFAULT_USER_COL].isin(train[DEFAULT_USER_COL].unique())]
    test = test[test[DEFAULT_ITEM_COL].isin(train[DEFAULT_ITEM_COL].unique())]
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    return NCFDataset(
        train_file=TRAIN_FILE,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        seed=SEED,
    )


def train_ncf(params, data):
    model = NCF(n_users=data.n_users, n_items=data.n_items, **params)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_ncf(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        users, items, preds = [], [], []
        item = list(train[DEFAULT_ITEM_COL].unique())
        for user in train[DEFAULT_USER_COL].unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))
        topk_scores = pd.DataFrame(
            data={
                DEFAULT_USER_COL: users,
                DEFAULT_ITEM_COL: items,
                DEFAULT_PREDICTION_COL: preds,
            }
        )
        merged = pd.merge(
            train, topk_scores, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="outer"
        )
        topk_scores = merged[merged[DEFAULT_RATING_COL].isnull()].drop(
            DEFAULT_RATING_COL, axis=1
        )
    # Remove temp files
    return topk_scores, t


def prepare_training_cornac(train, test):
    return cornac.data.Dataset.from_uir(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1).itertuples(index=False), seed=SEED
    )





def train_bpr(params, data):
    model = BPR(**params)
    with Timer() as t:
        model.fit(data)
    return model, t

def recommend_k_bpr(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            train,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t

def train_bivae(params, data):
    model = cornac.models.BiVAECF(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_bivae(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = predict_ranking(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def prepare_training_sar(train, test):
    return train


def train_sar(params, data):
    model = SAR(**params)
    model.set_index(data)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_sar(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def prepare_training_lightgcn(train, test):
    return ImplicitCF(train=train, test=test)


def train_lightgcn(params, data):
    hparams = prepare_hparams(**params)
    model = LightGCN(hparams, data)
    with Timer() as t:
        model.fit()
    return model, t


def recommend_k_lightgcn(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def rating_metrics_pyspark(test, predictions):
    rating_eval = SparkRatingEvaluation(test, predictions, **COL_DICT)
    return {
        "RMSE": rating_eval.rmse(),
        "MAE": rating_eval.mae(),
        "R2": rating_eval.exp_var(),
        "Explained Variance": rating_eval.rsquared(),
    }


def ranking_metrics_pyspark(test, predictions, k=DEFAULT_K):
    rank_eval = SparkRankingEvaluation(
        test, predictions, k=k, relevancy_method="top_k", **COL_DICT
    )
    return {
        "MAP": rank_eval.map(),
        "nDCG@k": rank_eval.ndcg_at_k(),
        "Precision@k": rank_eval.precision_at_k(),
        "Recall@k": rank_eval.recall_at_k(),
    }


def rating_metrics_python(test, predictions):
    return {
        "RMSE": rmse(test, predictions, **COL_DICT),
        "MAE": mae(test, predictions, **COL_DICT),
        "R2": rsquared(test, predictions, **COL_DICT),
        "Explained Variance": exp_var(test, predictions, **COL_DICT),
    }


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
    }
