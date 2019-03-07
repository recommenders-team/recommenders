# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com

import sys

MOVIELENS = '100k' # Select Movielens data size: 100k, 1m, 10m, or 20m.
TOPK      = 10     # Top k items to recommend.
TITLEN    = 45     # Truncation of titles in printing to screen.
SMPLS     = 10     # Number of observations to display.
MOVDISP   = 5      # Number of movies to display for a specific user.

sys.stdout.write("""======================
Microsoft Recommenders
======================

Welcome to a demo of the Microsoft open source Recommendations toolkit.
This is a Microsoft open source project though not a supported product.
Pull requests are most welcome.

This demo runs RBM, the Restricted Boltzmann Machine (RBM), a generative
neural network model, adapted as a recommendation algorithm, deployed on the
traditional Movielens benchmark dataset.

Press Enter to load the required Python modules: """)

# Import required libraries.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reco_utils.recommender.rbm.rbm          import RBM
from reco_utils.dataset.python_splitters     import numpy_stratified_split
from reco_utils.dataset.sparse               import AffinityMatrix
from reco_utils.dataset                      import movielens
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k
)

answer = input() # Wait for user.

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

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

  <User ID> <Item ID> [<Time>] [<Event Type>] [<Event Weight>].

Each observation is an interaction between a user and item (e.g., movie
watched on a streaming site or an item clicked on an e-commerce website).

The MovieLens dataset records movie ratings provided by viewers. The ratings
are treated as the event weights. The smaller of the available datasets is
used, consisting of 100K users.

Press Enter to download the dataset and show the first few observations: """)

data = movielens.load_pandas_df(size=MOVIELENS,
                                header=['UserID', 'MovieID',
                                        'Rating', 'Timestamp']
)

# Convert to 32-bit in order to reduce memory consumption.

data.loc[:, 'Rating'] = data['Rating'].astype(np.int32)

# Load the movie title index.

titles = pd.read_table('titles.txt',
                       sep='|',
                       header=None,
                       encoding="ISO-8859-1")
titles = titles.loc[:, 0:1]
titles.columns = ["MovieID", "MovieTitle"]

answer = input() # Wait for user.

sys.stdout.write("""
==============
Sample Ratings
==============

Below we illustrate the ratings that a number of users have provided for
specific movies. Note that the Rating column will be treated as the Event
Weight and we are not displaying the Time column. From the 100,000 events
in the dataset we will be partitioning the data into training and test
subsets. The model is built from the training dataset. 

""")

# Illustrative sample output. Rating is really a 1-5 integer and not a
# float so be sure to display as an integer rather than a
# float. Decide not to display Timestamp unless we convert to
# something understandable.

# TODO Replace truncated movie title with ... to be more informative.

smpl = pd.merge(data, titles, on="MovieID").sample(SMPLS)
smpl['MovieTitle'] = smpl['MovieTitle'].str[:TITLEN]
smpl['Rating'] = pd.to_numeric(smpl['Rating'], downcast='integer')
del smpl['Timestamp'] # Drop the column from printing.
print(smpl.to_string())

header ={
    "col_user"   : "UserID",
    "col_item"   : "MovieID",
    "col_rating" : "Rating",
}

# Use a sparse matrix representation rather than a pandas data frame
# for significant performance gain.

am = AffinityMatrix(DF=data, **header)
X  = am.gen_affinity_matrix()

# Contstruct the training and test datasets.

Xtr, Xtst = numpy_stratified_split(X)

print('\nTraining matrix size (users, movies) is:', Xtr.shape)
print('Testing matrix size is: ', Xtst.shape)

# Initialize the model class. Note that through random variation we
# can get a much better performing model with seed=1!

model = RBM(hidden_units   = 600,
            training_epoch = 30,
            minibatch_size = 60,
            keep_prob      = 0.9,
            with_metrics   = True,
#           seed           = 1,
)

sys.stdout.write("""
Press Enter to fit the model and to apply it to the dataset: """)
answer = input() # Wait for user.

### TODO HOW TO KEEP THE OUTPUT QUIET??????

train_time = model.fit(Xtr, Xtst)

### TODO HOW TO KEEP THE OUTPUT QUIET??????

top_k, test_time =  model.recommend_k_items(Xtst)

# Map the index back to original ids?????

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

    eval_map = map_at_k(data_true, data_pred,
                        col_user="UserID", col_item="MovieID", 
                        col_rating="Rating", col_prediction="prediction", 
                        relevancy_method="top_k", k=K)

    eval_ndcg = ndcg_at_k(data_true, data_pred,
                          col_user="UserID", col_item="MovieID", 
                          col_rating="Rating", col_prediction="prediction", 
                          relevancy_method="top_k", k=K)

    eval_precision = precision_at_k(data_true, data_pred,
                                    col_user="UserID", col_item="MovieID",
                                    col_rating="Rating",
                                    col_prediction="prediction", 
                                    relevancy_method="top_k", k=K)

    eval_recall = recall_at_k(data_true, data_pred,
                              col_user="UserID", col_item="MovieID", 
                              col_rating="Rating", col_prediction="prediction",
                              relevancy_method="top_k", k=K)

    
    df_result = pd.DataFrame(
        {   "Dataset"        : data_size,
            "K"              : TOPK,
            "MAP"            : eval_map,
            "nDCG@k"         : eval_ndcg,
            "Precision@k"    : eval_precision,
            "Recall@k"       : eval_recall,
            "Train time (s)" : time_train,
            "Test time (s)"  : time_test
        }, 
        index=[0]
    )
    
    return df_result

eval_100k= ranking_metrics(
    data_size  = "mv 100k",
    data_true  = test_df,
    data_pred  = top_k_df,
    time_train = train_time,
    time_test  = test_time,
    K          = TOPK
)

sys.stdout.write("""
The model took {:2.0f} seconds to fit and {:2.0f} seconds to rank.

Press Enter to continue to performance evalution: """.
                 format(train_time, test_time))
answer = input()

print("""
======================
Performance Evaluation
======================

We evaluate the perfomance of the model using typical recommendations model
performance criteria as provided by the Microsoft recommenders toolkit. The
following evaluation criteria are commonly used. 

Precision is the fraction of the K movies recommended that are relevant to the
user. Recall is the proportion of relevant items that are recommended. NDCG is
the Normalized Discounted Cumulative Gain which evaluates how well the 
predicted items for a user are ranked based on relevance. Finally, MAP is the
mean average precision, calcuated as the average precision for each user
normalised over all users.  MAP is generally a good discriminator between
models and is reported to be quite stable.

rbm_ref with @K={:2.0f}
Precision: {:4.2f} 
Recall:    {:4.2f}
NDCG:      {:4.2f}
MAP:       {:4.2f}
""".format(eval_100k.loc[0, "K"],
           eval_100k.loc[0, "Precision@k"],
           eval_100k.loc[0, "Recall@k"],
           eval_100k.loc[0, "nDCG@k"],
           eval_100k.loc[0, "MAP"]))
