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
import itertools

import time as tm

#from reco_utils.recommender.rbm.helperfunct import*

#ML Libraries and methods
import tensorflow as tf

from reco_utils.recommender.rbm.Mrbm_tensorflow import RBM
from reco_utils.dataset.python_splitters import python_stratified_split

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

#train/test split

train, test = python_stratified_split(ratings_df)

train.head()

test.head()

train.shape
test.shape


#--------------------Generate user affinity matrix--------------

#Generate index

#sort entries by user index
df = train.sort_values(by=["userID"])

#find unique user and item index
unique_users = df["userID"].unique()
unique_items = df["MovieId"].unique()

#total number of unique users and items
Nusers = len(unique_users)
Nitems = len(unique_items)

#create a dictionary to map unique users/items to hashed values to generate the matrix
map_users = {x:i for i, x in enumerate(unique_users)}
map_items = {x:i for i, x in enumerate(unique_items)}

#map back functions used to get back the original dataframe
map_back_users = {i:x for i, x in enumerate(unique_users)}
map_back_items = {i:x for i, x in enumerate(unique_items)}


unique_users.shape


#def gen_affinity_matrix(DF):

    df = test.copy()

    df.loc[:, 'hashedItems'] = df['MovieId'].map(map_items)
    df.loc[:, 'hashedUsers'] = df['userID'].map(map_users)

    #extract informations from the dataframe as an array. Note that we substract 1 from itm_id and usr_id
    #in order to map it to matrix format

    r_ = df['Rating']    #ratings
    itm_id = df['hashedItems']  #itm_id serving as columns
    usr_id = df['hashedUsers']  #usr_id serving as rows

    #check that all 3 vectors have the same dimensions
    assert((usr_id.shape[0]== r_.shape[0]) & (itm_id.shape[0] == r_.shape[0]))

    #generate a sparse matrix representation using scipy's coo_matrix and convert to array format
    RM = sparse.coo_matrix((r_, (usr_id, itm_id)), shape= (Nusers, Nitems)).toarray()

    #---------------------print the degree of sparsness of the matrix------------------------------

    zero   = (RM == 0).sum() # number of unrated items
    total  = RM.shape[0]*RM.shape[1] #number of elements in the matrix
    sparsness = zero/total *100 #Percentage of zeros in the matrix

    print('Matrix generated, sparsness %d' %sparsness,'%')

    return RM


#generate the ranking matrix. Movies that haven not been rated yet get a 0 rating.


X_train = gen_affinity_matrix(train)

X_test = gen_affinity_matrix(test)

X_train.shape
X_test.shape





#Distribution of values
zero_train  = (X_train == 0).sum()
total  = X_train.shape[0]*X_train.shape[1]

id =np.where(X_train !=0)

plt.hist(X_train[id], bins=5, density=1)

#Percentage of zeros in the matrix
zero_train/total *100

zero_test  = (X_test == 0).sum()

id_tst =np.where(X_test !=0)

plt.hist(X_test[id_tst], bins=5, density=1)

zero_test/total *100



#New method


header = {
        "col_user": "userID",
        "col_item": "MovieId",
        "col_rating": "Rating",
    }

model = RBM(hidden_units= 1000, keep_prob= .7, training_epoch = 10, **header)

param = model.fit(train)

#predict
top_k_df =  model.recommend_k_items(train)


#----------------------------------Create a user report with recommendations-----------------------------------------
usr_id = 0

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
