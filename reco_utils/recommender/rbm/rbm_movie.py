'''

Movie recommender with multinomial RBM v.0.2 (2018)
============================================================
Author: Mirco Milletari <mirco.milletari@microsoft.com>

Restricted Boltzmann machines are used to perform collaborative filtering over the Movielens dataset.
The model generates new ratings given the collected ratings. Model performance is also analysed over
different choices of the hyper parameter.

'''
import sys
sys.path.append("../../")

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from scipy import sparse


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import time as tm

#ML Libraries and methods
import tensorflow as tf

from reco_utils.recommender.rbm.Mrbm_tensorflow import RBM
from reco_utils.dataset.python_splitters import python_stratified_split, python_random_split

from reco_utils.dataset.rbm_splitters import splitter

#For interactive mode only
%load_ext autoreload
%autoreload 2


#Load the movielens dataset
dir = os.getcwd() #optain the absolute path
path1 = dir + '/reco_utils/recommender/rbm/movielens_data/movies.dat'

movies_df = pd.read_csv(path1, sep='::', header= None, engine= 'python', names =  ['MovieId', 'Title', 'Genre'] ) #movies dataset

#inspect first entries
movies_df.head()


#load the ratings
path2 = dir+ '/reco_utils/recommender/rbm/movielens_data/ratings.dat'
ratings_df = pd.read_csv(path2, sep='::', header=None, engine = 'python', names=['userID','MovieId','Rating','Timestamp'])

ratings_df.head()

#==========================
#Data processing
#==========================

#train/test split: comparing python_splitters

ratings_df['userID'].unique().shape
ratings_df['MovieId'].unique().shape


train, test = python_stratified_split(ratings_df)

train1, test1= python_random_split(ratings_df)


#total number of elements
train.shape
test.shape

train1.shape
test1.shape

#stratified
test['userID'].unique().shape
train['userID'].unique().shape
<<<<<<< HEAD

test['MovieId'].unique().shape
train['MovieId'].unique().shape

#random
test1['userID'].unique().shape
train1['userID'].unique().shape

test1['MovieId'].unique().shape
train1['MovieId'].unique().shape
=======

test['MovieId'].unique().shape
train['MovieId'].unique().shape

#random
test1['userID'].unique().shape
train1['userID'].unique().shape

test1['MovieId'].unique().shape
train1['MovieId'].unique().shape


train['MovieId'].shape
test['MovieId'].shape

#----------------------------------------------------------------------------------------------

header = {
        "col_user": "userID",
        "col_item": "MovieId",
        "col_rating": "Rating",
    }

data = splitter(DF = ratings_df, **header)

Xtr, Xtst, train_df, test_df, map = data.stratified_split()
<<<<<<< HEAD
=======


train_df.head()
>>>>>>> 067ef48d3a6cf61d910955dac2e2a792e5ae92f7

test_df.head()

<<<<<<< HEAD
train['MovieId'].shape
test['MovieId'].shape

#----------------------------------------------------------------------------------------------

header = {
        "col_user": "userID",
        "col_item": "MovieId",
        "col_rating": "Rating",
    }

data = splitter(DF = ratings_df, **header)

Xtr, Xtst = data.stratified_split()
>>>>>>> 3e56b7b20f83c9080d1b2ccfc68c834e27ebc64d


train_df.head()

test_df.head()

=======
>>>>>>> 067ef48d3a6cf61d910955dac2e2a792e5ae92f7
#Check the number of elements
(Xtr !=0).sum()
(Xtst !=0).sum()


#Distribution of values
zero_train  = (Xtr == 0).sum()
total  = Xtr.shape[0]*Xtr.shape[1]

id =np.where(Xtr !=0)

plt.hist(Xtr[id], bins=5, density=1)

#Percentage of zeros in the matrix
zero_train/total *100

zero_test  = (Xtst == 0).sum()

id_tst =np.where(Xtst !=0)

plt.hist(Xtst[id_tst], bins=5, density=1)

zero_test/total *100

#===========================
#Train the model
#===========================

model = RBM(hidden_units= 500, save_model=True, keep_prob= .7, training_epoch = 10, **header)
<<<<<<< HEAD

model.fit(Xtr,Xtst)



#predict
<<<<<<< HEAD
=======
top_k_df =  model.recommend_k_items(Xtr)
=======

model.fit(Xtr,Xtst)



#predict
>>>>>>> 3e56b7b20f83c9080d1b2ccfc68c834e27ebc64d
top_k = 10
results, top_items =  model.recommend_k_items(Xtr, map)

results.head(10)
<<<<<<< HEAD
=======
>>>>>>> 067ef48d3a6cf61d910955dac2e2a792e5ae92f7
>>>>>>> 3e56b7b20f83c9080d1b2ccfc68c834e27ebc64d

#Check if the mapping is correct
[ map[1][top_items[0][i]] for i in range(10) ]


#----------------------------------Create a user report with recommendations-----------------------------------------
<<<<<<< HEAD
=======
<<<<<<< HEAD
usr_id = 0
=======
>>>>>>> 067ef48d3a6cf61d910955dac2e2a792e5ae92f7
>>>>>>> 3e56b7b20f83c9080d1b2ccfc68c834e27ebc64d

usr_mv_like = np.where(vp[usr_id]==2)
mv_id = (np.asanyarray(usr_mv_like)+1).flatten()

MVI= np.in1d(movies_df['ItemID'].values, mv_id)

sel_movie = movies_df[MVI]
sel_movie['reco score'] = pvh[usr_id,usr_mv_like,1].flatten()



sel_movie.sort_values(['reco score'], ascending = False).head(10)

merged_df = pd.merge(movies_df, ratings_df[['UserID', 'ItemID','Rating']], on='ItemID')
merged_df[ merged_df['UserID'] ==1 ]



mv_unseen = np.where(RX[usr_id]==0)
mv_id_un= np.intersect1d(usr_mv_like, mv_unseen)
mv_id_unseen = mv_id_un+1

MVI_unseen= np.in1d(movies_df['ItemID'].values, mv_id_unseen)

sel_unseen_movie = movies_df[MVI_unseen]

sel_unseen_movie['reco score'] = pvh[usr_id,mv_id_un,1].flatten()

sel_unseen_movie.sort_values(['reco score'], ascending = False).head(10)
