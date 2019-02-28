# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com

print("""======================
Microsoft Recommenders
======================

Welcome to a demo of the Microsoft open source Recommendations toolkit.
This is not a Microsoft product and hence has no official support. It is 
a Microsoft open source project, so pull requests are most welcome.

This demo runs SAR, the smart adaptive recommendation algorithm on the
traditional Movielens benchmark dataset.

Loading the required Python modules and reporting versions ...
""")

# Import required libraries.

import sys
import time
import os
import itertools
import pandas as pd
import warnings
import recutils
import imdb
import urllib.request

from shutil import copyfile

from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

sys.stdout.write("""
Press Enter to continue: """)
answer = input()

sys.stdout.write("""
=============
SAR Algorithm
=============

SAR is a fast smart adaptive algorithm for personalized recommendations
based on user history using collaborative filtering. It produces easily
explainable and interpretable recommendations and handles "cold item"
and "semi-cold user" scenarios. Data schema is:

  <User ID>, <Item ID>, <Time>, [<Event Type>, [<Event Weight>]].

Each observation is an interaction between a user and item (e.g., movie
watched or item clicked on an e-commerce website).

The MovieLens dataset is used here. It records interactions of Users providing
Ratings to Movies (movie ratings are used as the event weight). The smaller of
the avaiable datasets is used, consisting of 100K users.

Press Enter to load the dataset and show the first few observations: """)

# Top k items to recommend.

TOPK = 10

# Select Movielens data size: 100k, 1m, 10m, or 20m.

MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId','MovieId','Rating','Timestamp']
)

# Load the movie title index.

titles = pd.read_table('titles.txt', sep='|', header=None, encoding = "ISO-8859-1")
titles = titles.loc[:, 0:1]
titles.columns = ["MovieId", "MovieTitle"]

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

print(pd.merge(data, titles, on="MovieId").sample(10).to_string())

train, test = python_random_split(data)

header ={
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

# The reverse of the dictionary above - array index to actual ID.

index2user = dict(enumerate_users_2)
index2item = dict(enumerate_items_2)

# We need to index the train and test sets for SAR matrix operations to work.

model.set_index(unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item)

preprocess_time = time.time() - start_time

start_time = time.time()

# Suppress the SettingWithCopyWarning for now.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train)

train_time = time.time() - start_time + preprocess_time
# print("""
# Took {} seconds for training.
# """.format(train_time))

start_time = time.time()

# Suppress the SettingWithCopyWarning for now.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    topk = model.recommend_k_items(test)

test_time = time.time() - start_time
# print("""
# Took {} seconds for prediction.
# """.format(test_time))

# TODO: remove this call when the model returns same type as input

topk['UserId'] = pd.to_numeric(topk['UserId'])
topk['MovieId'] = pd.to_numeric(topk['MovieId'])
topk['prediction'] = pd.to_numeric(topk['prediction'])


sys.stdout.write("""
Press Enter to fit the model (<1s) and apply to the testing dataset: """)
answer = input()

print("""
==================
Sample Predictions
==================

For a random sample of users from the testing dataset we list the model's 
prediction of their rating of a particular movie. The predicted ratings 
are used for ranking their preferences rather than specific ratings, hence
we see some values beyond the 1-5 range of the ratings. The rankings are used
to suggest several (K=10, perhaps) movies that the user has not previously
seen but are likely to be highly rated by that user.
""")

smpl = topk.sample(10)
smpl['prediction'] = round(smpl['prediction'])
print(pd.merge(smpl, titles, on="MovieId").to_string())

sys.stdout.write("""
Press Enter to view the movies watched and recommended for a random user: """)

DISP = 5

u1  = smpl["UserId"].iloc[0]
u1m = data.ix[(data['UserId'] == u1)].sort_values('Rating', ascending=False)[:DISP]
u1p = topk.ix[(topk['UserId'] == u1)].sort_values('prediction', ascending=False)[:DISP]

answer = input()

sys.stdout.write("""
=======================
Show Random User Movies
=======================

For the first user above (originally chosen at random) we list below their
top 5 rated movies and then the movies that are recommended for them.
This user has actually rated {} movies in total.

""".format(len(data.ix[(data['UserId'] == u1)])))

print(pd.merge(u1m, titles, on="MovieId").to_string())

print ("")

print(pd.merge(u1p, titles, on="MovieId").to_string())

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
    
    for i in range(DISP):
        title = u1mt[i]
        movie = ia.search_movie(title)[0]
        ia.update(movie)
        cover = movie.get('cover url')
        dst = "m{}.jpg".format(i)
        if cover is not None:
            urllib.request.urlretrieve(cover, dst)
        else:
            copyfile("na.jpg", dst)

    for i in range(DISP):
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

eval_precision = precision_at_k(test, topk, col_user="UserId", col_item="MovieId", 
                                col_rating="Rating", col_prediction="prediction", 
                                relevancy_method="top_k", k=TOPK)

eval_recall = recall_at_k(test, topk, col_user="UserId", col_item="MovieId", 
                          col_rating="Rating", col_prediction="prediction", 
                          relevancy_method="top_k", k=TOPK)

answer = input()

print("""
======================
Performance Evaluation
======================

We evaluate the perfomance of the model using typical recommendations model
performance criteria as provided by the Microsoft recommenders toolkit. The
following evaluation criteria are the common ones. 

TODO: brief summary of each.

Model: {}
Top K: {:d}
MAP:         {:4.2f}
NDCG:        {:4.2f}
Precision@K: {:4.2f} 
Recall@K:    {:4.2f}
""".format(model.model_str, TOPK, eval_map, eval_ndcg,
                               eval_precision, eval_recall))
