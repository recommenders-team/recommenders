# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com

print("""======================
Microsoft Recommenders
======================

Welcome to a demo of the Microsoft open source Recommendations toolkit.
This is not a Microsoft product and hence has no official support. It is 
a Microsoft open source project, so pull requests are most welcome.

This demo runs RBM, the Restricted Boltzmann Machine (RBM), a generative
neural network model, adapted as a recommendation algorithm, deployed on the
traditional Movielens benchmark dataset.

Now loading the required Python modules and will report on library versions ...
""")

# Import required libraries.

import sys
#import time
import os
#import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import warnings
#import recutils
#import imdb
#import urllib.request

#from shutil import copyfile

#from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
#from reco_utils.dataset.python_splitters import python_random_split

from reco_utils.recommender.rbm.rbm import RBM
from reco_utils.dataset.numpy_splitters import numpy_stratified_split
from reco_utils.dataset.sparse import AffinityMatrix

from reco_utils.dataset import movielens
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

sys.stdout.write("""
Press Enter to continue: """)
answer = input()

sys.stdout.write("""
=============
RBM Algorithm
=============

RBM generates ratings for a user/item (i.e., movie) pair using a collaborative
filtering based approach. While matrix factorization methods learn how to
reproduce an instance of the user/item affinity matrix, the RBM learns the
underlying probability distribution. This has several advantages related to
generalizability, training stability, and fast training on GPUs.

The data schema for the training dataset is:

  <User ID>, <Item ID>, <Time>, [<Event Type>, [<Event Weight>]].

Each observation is an interaction between a user and item (e.g., movie
watched or item clicked on an e-commerce website).

The MovieLens dataset is used here. It records interactions of Users providing
Ratings to Movies (movie ratings are used as the event weight). The smaller
dataset is used for this demonstration consisting of 100K users.

Press Enter to download the dataset and show the first few observations: """)

# Top k items to recommend.

TOPK = 10

# Select Movielens data size: 100k, 1m, 10m, or 20m.

MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['userID','movieID','rating','timestamp']
)

# Convert to 32-bit in order to reduce memory consumption.

data.loc[:, 'rating'] = data['rating'].astype(np.int32)

# Load the movie title index.

titles = pd.read_table('titles.txt',
                       sep='|',
                       header=None,
                       encoding="ISO-8859-1")
titles = titles.loc[:, 0:1]
titles.columns = ["movieID", "movieTitle"]

answer = input()
print("""
==============
Sample Ratings
==============

Below we show the ratings that a number of users have provided for specific
movies. The order of the columns does not particularly matter, and so we note
that Rating is the Event Weight and Timestamp is the Time column. From the 
100,000 events in the dataset we will be partitioning the data into training
and testing subsets. The model is built from the training dataset. 
""")

# TODO Replace truncated movie title with ... to be more informative.

smpl = pd.merge(data, titles, on="movieID").sample(10)
smpl['movieTitle'] = smpl['movieTitle'].str[:35]
print(smpl.to_string())

header ={
    "col_user": "userID",
    "col_item": "movieID",
    "col_rating": "rating",
}

# TO EXPLAIN


# Instantiate the sparse matrix generation.

am = AffinityMatrix(DF = data, **header)

# Obtain the sparse matrix.

X = am.gen_affinity_matrix()

Xtr, Xtst = numpy_stratified_split(X)

print('\nTraining matrix size (users, movies) is:', Xtr.shape)
print('Testing matrix size is: ', Xtst.shape)


#First we initialize the model class

model = RBM(hidden_units=600,
            training_epoch=30,
            minibatch_size=60,
            keep_prob=0.9,
            with_metrics=True)

sys.stdout.write("""
Press Enter to continue with the model fitting: """)
answer = input()

#Model Fit
### TODO HOW TO KEEP THE OUTPUT QUIET??????
train_time= model.fit(Xtr, Xtst)

#number of top score elements to be recommended  
K = 10

#Model prediction on the test set Xtst. 
### TODO HOW TO KEEP THE OUTPUT QUIET??????
top_k, test_time =  model.recommend_k_items(Xtst)

top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
test_df = am.map_back_sparse(Xtst, kind = 'ratings')



top_k_df.head(10)

def ranking_metrics(
    data_size,
    data_true,
    data_pred,
    time_train,
    time_test,
    K
):

    eval_map = map_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                    col_rating="rating", col_prediction="prediction", 
                    relevancy_method="top_k", k= K)

    eval_ndcg = ndcg_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                      col_rating="rating", col_prediction="prediction", 
                      relevancy_method="top_k", k= K)

    eval_precision = precision_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                               col_rating="rating", col_prediction="prediction", 
                               relevancy_method="top_k", k= K)

    eval_recall = recall_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                          col_rating="rating", col_prediction="prediction", 
                          relevancy_method="top_k", k= K)

    
    df_result = pd.DataFrame(
        {   "Dataset": data_size,
            "K": K,
            "MAP": eval_map,
            "nDCG@k": eval_ndcg,
            "Precision@k": eval_precision,
            "Recall@k": eval_recall,
            "Train time (s)": time_train,
            "Test time (s)": time_test
        }, 
        index=[0]
    )
    
    return df_result

eval_100k= ranking_metrics(
    data_size = "mv 100k",
    data_true =test_df,
    data_pred =top_k_df,
    time_train=train_time,
    time_test =test_time,
    K =10)

sys.stdout.write("""
Press Enter to continue to performance evalution: """)
answer = input()

print("""
======================
Performance Evaluation
======================

We evaluate the perfomance of the model using typical recommendations model
performance criteria as provided by the Microsoft recommenders toolkit. The
following evaluation criteria are the common ones. 

TODO: brief summary of each.

Model: rbm_ref
Top K: {:2.0f}
MAP:         {:4.2f}
NDCG:        {:4.2f}
Precision@K: {:4.2f} 
Recall@K:    {:4.2f}
""".format(eval_100k.loc[0, "K"],
                               eval_100k.loc[0, "MAP"],
                               eval_100k.loc[0, "nDCG@k"],
                               eval_100k.loc[0, "Precision@k"],
                               eval_100k.loc[0, "Recall@k"]))
