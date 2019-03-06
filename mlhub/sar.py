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

This demo runs SAR, the smart adaptive recommendation algorithm on the
traditional MovieLens benchmark dataset which is freely available from
https://grouplens.org/datasets/movielens/.

Press Enter to load the required Python modules: """)

# Import required libraries.

import os
import time
import itertools

import numpy as np
import pandas as pd

import warnings
import recutils
import imdb
import urllib.request

from shutil import copyfile

from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
from reco_utils.dataset                        import movielens
from reco_utils.dataset.python_splitters       import python_random_split
from reco_utils.evaluation.python_evaluation   import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k
)

answer = input() # Wait for user.

print("\nSystem version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

sys.stdout.write("""
=============
SAR Algorithm
=============

SAR is a fast smart adaptive algorithm for personalized recommendations based
on user history using collaborative filtering. It produces easily explainable
and interpretable recommendations and handles "cold item" and "semi-cold user"
scenarios. The training data schema is:

  <User ID> <Item ID> <Time> [<Event Type>] [<Event Weight>].

Each observation is an interaction between a user and item (e.g., a movie
watched on a streaming site or an item clicked on an e-commerce website).

The MovieLens dataset records movie ratings provided by viewers. The ratings
are treated as the event weights. The smaller of the available datasets is
used, consisting of 100K users.

Press Enter to load the dataset and show the first few observations: """)

data = movielens.load_pandas_df(size   = MOVIELENS,
                                header = ['UserId',
                                          'MovieId',
                                          'Rating',
                                          'Timestamp']
)

# Convert float precision to 32-bit to reduce memory consumption.

data.loc[:, 'Rating'] = data['Rating'].astype(np.float32)

# Load the movie title index.

titles = pd.read_table('titles.txt',
                       sep      = '|',
                       header   = None,
                       encoding = "ISO-8859-1")
titles = titles.loc[:, 0:1]
titles.columns = ["MovieId", "MovieTitle"]

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

smpl = pd.merge(data, titles, on="MovieId").sample(SMPLS)
smpl['MovieTitle'] = smpl['MovieTitle'].str[:TITLEN]
smpl['Rating'] = pd.to_numeric(smpl['Rating'], downcast='integer')
del smpl['Timestamp'] # Drop the column from printing.
print(smpl.to_string())

# Create train and test datasets.

train, test = python_random_split(data)

# Create a model object.

header = {
    "col_user"      : "UserId",
    "col_item"      : "MovieId",
    "col_rating"    : "Rating",
    "col_timestamp" : "Timestamp",
}

model = SARSingleNode(remove_seen            = True,
                      similarity_type        = "jaccard", 
                      time_decay_coefficient = 30,
                      time_now               = None,
                      timedecay_formula      = True,
                      **header
)

start_time = time.time()
model.fit(train)
train_time = time.time() - start_time

start_time = time.time()
topk = model.recommend_k_items(test)
test_time = time.time() - start_time

# TODO: remove this call when the model returns same type as input

topk['UserId'] = pd.to_numeric(topk['UserId'])
topk['MovieId'] = pd.to_numeric(topk['MovieId'])

sys.stdout.write("""
Press Enter to fit the model (<1s) and to apply it to the test dataset: """)
answer = input() # Wait for user.

sys.stdout.write("""
==================
Sample Predictions
==================

For a random sample of users from the test dataset we list the model's
prediction of their rating of a particular movie. The predicted ratings 
are used in ranking their preferences rather than as specific ratings, hence
we see some values beyond the 1-5 range. The rankings are used to suggest
several (K=10) movies that the user has not previously seen but are likely to
be highly rated by that user.

""")

smpl = topk.sample(SMPLS)
smpl['Predict'] = round(smpl['prediction'],1)
smpl = smpl[['UserId', 'MovieId', 'Predict']] # Reorder columns.
smpl = pd.merge(smpl, titles, on="MovieId")
smpl['MovieTitle'] = smpl['MovieTitle'].str[:TITLEN]
print(smpl.to_string())

sys.stdout.write("""
Press Enter to view the movies watched and recommended for a random user: """)
answer = input() # Wait for user.

u1 = smpl["UserId"].iloc[0] # Random user.

sys.stdout.write("""
=======================
Show Random User Movies
=======================

For the first user above (originally chosen at random) we list below their
top 5 rated movies and then the movies that are recommended for them.
This user has actually rated {} movies in total.

""".format(len(data.ix[(data['UserId'] == u1)])))

u1m = data.ix[(data['UserId'] == u1)].sort_values('Rating',
                                                  ascending=False)[:MOVDISP]
u1p = topk.ix[(topk['UserId'] == u1)].sort_values('prediction',
                                                  ascending=False)[:MOVDISP]
smpl = u1m
smpl['Rating'] = pd.to_numeric(smpl['Rating'], downcast='integer')
smpl = smpl[['UserId', 'MovieId', 'Rating']] # Drop Timestamp column.
smpl = pd.merge(smpl, titles, on="MovieId")
smpl['MovieTitle'] = smpl['MovieTitle'].str[:TITLEN]
print(smpl.to_string(), "\n")

smpl = u1p
smpl['Predict'] = round(smpl['prediction'], 1)
smpl = smpl[['UserId', 'MovieId', 'Predict']] # Reorder the columns.
smpl = pd.merge(smpl, titles, on="MovieId")
smpl['MovieTitle'] = smpl['MovieTitle'].str[:TITLEN]
print(smpl.to_string())

sys.stdout.write("""
We can generate a visual to show the top 5 rated movies and the movies
recommended for the user. This requires downloading images from Amazon
which can take a minute or two. 

Shall we construct the visual? [Y|n]: """)

answer = input()

if len(answer) == 0 or answer.lower()[0] != "n":

    u1mt = pd.merge(u1m, titles, on="MovieId").loc[:,'MovieTitle']
    u1pt = pd.merge(u1p, titles, on="MovieId").loc[:,'MovieTitle']

    ia = imdb.IMDb()
    
    for i in range(MOVDISP):
        title = u1mt[i]
        movie = ia.search_movie(title)[0]
        ia.update(movie)
        cover = movie.get('cover url')
        dst = "m{}.jpg".format(i)
        if cover is not None:
            urllib.request.urlretrieve(cover, dst)
        else:
            copyfile("na.jpg", dst)

    for i in range(MOVDISP):
        title = u1pt[i]
        movie = ia.search_movie(title)[0]
        ia.update(movie)
        cover = movie.get('cover url')
        dst = "p{}.jpg".format(i)
        if cover is not None:
            urllib.request.urlretrieve(cover, dst)
        else:
            copyfile("na.jpg", dst)

    recutils.plot_recommendations()

else:
    
    sys.stdout.write("""
We will display a sample image that was generated previously.
""")

    os.system("eom sample.png")

sys.stdout.write("""
Press Enter to continue on to the model performance evaluation: """)

eval_map = map_at_k(test, topk, col_user="UserId", col_item="MovieId", 
                    col_rating="Rating", col_prediction="prediction", 
                    relevancy_method="top_k", k=TOPK)

eval_ndcg = ndcg_at_k(test, topk, col_user="UserId", col_item="MovieId", 
                      col_rating="Rating", col_prediction="prediction", 
                      relevancy_method="top_k", k=TOPK)

eval_precision = precision_at_k(test, topk, col_user="UserId",
                                col_item="MovieId", 
                                col_rating="Rating",
                                col_prediction="prediction", 
                                relevancy_method="top_k", k=TOPK)

eval_recall = recall_at_k(test, topk, col_user="UserId", col_item="MovieId", 
                          col_rating="Rating", col_prediction="prediction", 
                          relevancy_method="top_k", k=TOPK)

answer = input()

sys.stdout.write("""
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
models and is quite stable.

{} with @K={:d}
Precision: {:4.2f} 
Recall:    {:4.2f}
NDCG:      {:4.2f}
MAP:       {:4.2f}

""".format(model.model_str, TOPK,
           eval_precision,
           eval_recall,
           eval_ndcg,
           eval_map))
