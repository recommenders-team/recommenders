import sys
sys.path.append("../")
import os
import json
import shutil
import tempfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType
from fastai.collab import EmbeddingDotBias, collab_learner, CollabDataBunch, load_learner

from reco_utils.common.general_utils import get_number_processors
from reco_utils.common.timer import Timer
from reco_utils.common.gpu_utils import get_cuda_version, get_cudnn_version
from reco_utils.common.spark_utils import start_or_get_spark
from reco_utils.dataset import movielens
from reco_utils.dataset.sparse import AffinityMatrix
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.recommender.surprise.surprise_utils import surprise_trainset_to_df
from reco_utils.recommender.fastai.fastai_utils import hide_fastai_progress_bar, cartesian_product, score
from reco_utils.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.evaluation.python_evaluation import rmse, mae, rsquared, exp_var


USER_COL = "UserId"
ITEM_COL = "MovieId"
RATING_COL = "Rating"
TIMESTAMP_COL = "Timestamp"
PREDICTION_COL = "prediction"
SEED = 77


def prepare_training_als(train):
    schema = StructType(
    (
        StructField(USER_COL, IntegerType()),
        StructField(ITEM_COL, IntegerType()),
        StructField(RATING_COL, FloatType()),
        StructField(TIMESTAMP_COL, LongType()),
    )
    )
    spark = start_or_get_spark()
    return spark.createDataFrame(train, schema)


def train_als(params, data):
    symbol = ALS(**params)
    with Timer() as t:
        model = symbol.fit(data)
    return model, t


def prepare_metrics_als(train, test):
    schema = StructType(
    (
        StructField(USER_COL, IntegerType()),
        StructField(ITEM_COL, IntegerType()),
        StructField(RATING_COL, FloatType()),
        StructField(TIMESTAMP_COL, LongType()),
    )
    )
    spark = start_or_get_spark()
    return prepare_training_als(train), spark.createDataFrame(test, schema)


def predict_als(model, test):
    with Timer() as t:
        preds = model.transform(test)
    return preds, t


def recommend_k_als(model, test, train):
    with Timer() as t:
        # Get the cross join of all user-item pairs and score them.
        users = train.select(USER_COL).distinct()
        items = train.select(ITEM_COL).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)

        # Remove seen items.
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            train.alias("train"),
            (dfs_pred[USER_COL] == train[USER_COL]) & (dfs_pred[ITEM_COL] == train[ITEM_COL]),
            how='outer'
        )
        top_k_scores = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train." + RATING_COL].isNull()) \
            .select('pred.' + USER_COL, 'pred.' + ITEM_COL, 'pred.' + PREDICTION_COL)
    return top_k_scores, t


def prepare_training_svd(train):
    reader = surprise.Reader('ml-100k', rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(train.drop(TIMESTAMP_COL, axis=1), reader=reader).build_full_trainset()


def train_svd(params, data):
    model = surprise.SVD(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def predict_svd(model, test):
    with Timer() as t:
        preds = [model.predict(row[USER_COL], row[ITEM_COL], row[RATING_COL])
                       for (_, row) in test.iterrows()]
        preds = pd.DataFrame(preds)
        preds = preds.rename(index=str, columns={"uid": USER_COL, 
                                                 "iid": ITEM_COL,
                                                 "est": PREDICTION_COL})
        preds = preds.drop(["details", "r_ui"], axis="columns")
        for col in [USER_COL, ITEM_COL]:
            preds[col] = preds[col].astype(int)
    return preds, t


def recommend_k_svd(model, test, train):
    with Timer() as t:
        preds_lst = []
        for user in train[USER_COL].unique():
            for item in train[ITEM_COL].unique():
                preds_lst.append([user, item, model.predict(user, item).est])
        top_k_scores = pd.DataFrame(data=preds_lst, columns=[USER_COL, ITEM_COL, PREDICTION_COL])
        merged = pd.merge(train, top_k_scores, on=[USER_COL, ITEM_COL], how="outer")
        top_k_scores = merged[merged[RATING_COL].isnull()].drop(RATING_COL, axis=1)
    return top_k_scores, t


def prepare_training_fastai(train):
    data = train.copy()
    data[USER_COL] = data[USER_COL].astype('str')
    data[ITEM_COL] = data[ITEM_COL].astype('str')
    data = CollabDataBunch.from_df(data, user_name=USER_COL, item_name=ITEM_COL, rating_name=RATING_COL)
    return data


def train_fastai(params, data):
    model = collab_learner(data, 
                           n_factors=params["n_factors"],
                           y_range=params["y_range"],
                           wd=params["wd"]
                          )
    with Timer() as t:
        model.fit_one_cycle(cyc_len=params["epochs"], max_lr=params["max_lr"])
    return model, t


def prepare_metrics_fastai(train, test):
    data = test.copy()
    data[USER_COL] = data[USER_COL].astype('str')
    data[ITEM_COL] = data[ITEM_COL].astype('str')
    return train, data


def predict_fastai(model, test):
    with Timer() as t:
        preds = score(model, 
                      test_df=test, 
                      user_col=USER_COL, 
                      item_col=ITEM_COL, 
                      prediction_col=PREDICTION_COL)
    return preds, t


def recommend_k_fastai(model, test, train):
    with Timer() as t: 
        total_users, total_items = model.data.train_ds.x.classes.values()
        total_items = total_items[1:]
        total_users = total_users[1:]
        test_users = test[USER_COL].unique()
        test_users = np.intersect1d(test_users, total_users)
        users_items = cartesian_product(test_users, total_items)
        users_items = pd.DataFrame(users_items, columns=[USER_COL, ITEM_COL])
        training_removed = pd.merge(users_items, train.astype(str), on=[USER, ITEM], how='left')
        training_removed = training_removed[training_removed[RATING].isna()][[USER, ITEM]]
        top_k_scores = score(model, 
                             test_df=training_removed,
                             user_col=USER_COL, 
                             item_col=ITEM_COL, 
                             prediction_col=PREDICTION_COL, 
                             top_k=TOP_K)
    return top_k_scores, t


def prepare_training_ncf(train):
    data = NCFDataset(train=train, 
                      col_user=USER_COL,
                      col_item=ITEM_COL,
                      col_rating=RATING_COL,
                      col_timestamp=TIMESTAMP_COL,
                      seed=SEED)
    return data


def train_ncf(params, data):
    model = NCF(n_users=data.n_users, n_items=data.n_items, **params)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_ncf(model, test, train):
    with Timer() as t: 
        users, items, preds = [], [], []
        item = list(train[ITEM_COL].unique())
        for user in train[USER_COL].unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))
        top_k_scores = pd.DataFrame(data={USER_COL: users, ITEM_COL:items, PREDICTION_COL:preds})
        merged = pd.merge(train, top_k_scores, on=[USER_COL, ITEM_COL], how="outer")
        top_k_scores = merged[merged[RATING_COL].isnull()].drop(RATING_COL, axis=1)
    return top_k_scores, t


def train_sar(params, data):
    model = SARSingleNode(**params)
    model.set_index(data)    
    with Timer() as t:
        model.fit(data)
    return model, t
    

def recommend_k_sar(model, test, train):
    with Timer() as t:
        top_k_scores = model.recommend_k_items(test)
    return top_k_scores, t


def rating_metrics_pyspark(test, predictions):
    rating_eval = SparkRatingEvaluation(test, 
                                        predictions, 
                                        col_user=USER_COL, 
                                        col_item=ITEM_COL, 
                                        col_rating=RATING_COL, 
                                        col_prediction=PREDICTION_COL)
    return {
        "RMSE": rating_eval.rmse(),
        "MAE": rating_eval.mae(),
        "R2": rating_eval.exp_var(),
        "Explained Variance": rating_eval.rsquared()
    }
    
    
def ranking_metrics_pyspark(test, predictions, k=10):
    rank_eval = SparkRankingEvaluation(test, 
                                       predictions, 
                                       k=k, 
                                       col_user=USER_COL, 
                                       col_item=ITEM_COL, 
                                       col_rating=RATING_COL, 
                                       col_prediction=PREDICTION_COL, 
                                       relevancy_method="top_k")
    return {
        "MAP": rank_eval.map_at_k(),
        "nDCG@k": rank_eval.ndcg_at_k(),
        "Precision@k": rank_eval.precision_at_k(),
        "Recall@k": rank_eval.recall_at_k()
    }
    
    
def rating_metrics_python(test, predictions):
    cols = {
        "col_user": USER_COL, 
        "col_item": ITEM_COL, 
        "col_rating": RATING_COL, 
        "col_prediction": PREDICTION_COL
    }
    return {
        "RMSE": rmse(test, predictions, **cols),
        "MAE": mae(test, predictions, **cols),
        "R2": rsquared(test, predictions, **cols),
        "Explained Variance": exp_var(test, predictions, **cols)
    }
    
    
def ranking_metrics_python(test, predictions, k=10):
    cols = {
        "col_user": USER_COL, 
        "col_item": ITEM_COL, 
        "col_rating": RATING_COL, 
        "col_prediction": PREDICTION_COL
    }
    return {
        "MAP": map_at_k(test, predictions, k=k, **cols),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **cols),
        "Precision@k": precision_at_k(test, predictions, k=k, **cols),
        "Recall@k": recall_at_k(test, predictions, k=k, **cols)
    }