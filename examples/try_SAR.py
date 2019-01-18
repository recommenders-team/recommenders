

import sys
sys.path.append("../../")
import time
import os
import itertools
import pandas as pd
#import papermill as pm

from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

TOP_K = 10

# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId','MovieId','Rating','Timestamp']
)

data.head()

train, test = python_random_split(data)

header = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_timestamp": "Timestamp",
}

model = SARSingleNode(
    remove_seen=True, similarity_type="jaccard",
    time_decay_coefficient=30, time_now=None, timedecay_formula=True, **header
)

start_time = time.time()

unique_users = data["UserId"].unique()
unique_items = data["MovieId"].unique()
enumerate_items_1, enumerate_items_2 = itertools.tee(enumerate(unique_items))
enumerate_users_1, enumerate_users_2 = itertools.tee(enumerate(unique_users))

item_map_dict = {x: i for i, x in enumerate_items_1}
user_map_dict = {x: i for i, x in enumerate_users_1}
# The reverse of the dictionary above - array index to actual ID
index2user = dict(enumerate_users_2)
index2item = dict(enumerate_items_2)

# We need to index the train and test sets for SAR matrix operations to work
model.set_index(unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item)

preprocess_time = time.time() - start_time

start_time = time.time()

model.fit(train)

train_time = time.time() - start_time + preprocess_time
print("Took {} seconds for training.".format(train_time))

start_time = time.time()

top_k = model.recommend_k_items(test)

test_time = time.time() - start_time
print("Took {} seconds for prediction.".format(test_time))

# TODO: remove this call when the model returns same type as input
top_k['UserId'] = pd.to_numeric(top_k['UserId'])
top_k['MovieId'] = pd.to_numeric(top_k['MovieId'])

#display(top_k.head())

eval_map = map_at_k(test, top_k, col_user="UserId", col_item="MovieId",
                    col_rating="Rating", col_prediction="prediction",
                    relevancy_method="top_k", k=TOP_K)

eval_ndcg = ndcg_at_k(test, top_k, col_user="UserId", col_item="MovieId",
                      col_rating="Rating", col_prediction="prediction",
                      relevancy_method="top_k", k=TOP_K)

eval_precision = precision_at_k(test, top_k, col_user="UserId", col_item="MovieId",
                                col_rating="Rating", col_prediction="prediction",
                                relevancy_method="top_k", k=TOP_K)

eval_recall = recall_at_k(test, top_k, col_user="UserId", col_item="MovieId",
                          col_rating="Rating", col_prediction="prediction",
                          relevancy_method="top_k", k=TOP_K)

print("Model:\t" + model.model_str,
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')