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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import time as tm

from reco_utils.recommender.rbm.helperfunct import*

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

#generate the ranking matrix. Movies that haven not been rated yet get a 0 rating.

#generate using scipy method
#st1= tm.time()
X_train = gen_ranking_matrix(train)
X_test = gen_ranking_matrix(test)

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


model = RBM(hidden_units= 10, keep_prob= 1, training_epoch = 10,**header)
model.fit(train)





#==========================
#Start the training
#==========================

def main(Xtr, Xtst, rating_scale, keep_prob, N_hidden, n_epochs, minibatch_size, learning_rate, save):

    #Network topology
    m, Nv= Xtr.shape #m= #of users,  number of visible units Nv = number of movies

    tf.reset_default_graph()  #to be able to rerun the model without overwriting tf variables

    reco= rbm.RBM(Nv, N_hidden, rating_scale, keep_prob) #Initialize the RBM.

    if save == True:
        saver = tf.train.Saver()

    init_g = tf.global_variables_initializer()

    #Open a TF session
    with tf.Session() as sess:

        reco.set_session(sess)
        sess.run(init_g)

        Mse_train, Mse_tst = reco.train(Xtr, Xtst, epochs= n_epochs, alpha = learning_rate, minibatch_size = minibatch_size)
        vp, pvh = reco.predict(Xtr)

        if save == True:
            saver.save(sess, 'saver/rbm_model_saver.ckpt')

    plt.plot(Mse_train, label= 'train')
    plt.ylabel('msr_error', size= 'x-large')
    plt.xlabel('epochs', size = 'x-large')
    plt.legend(ncol=1)

    sess.close()

    return Mse_train, Mse_tst, vp, pvh

Mse_train, Mse_test, vp, pvh = main(B_tr,B_tst, rating_scale= 2, keep_prob= 1,  N_hidden= 10, n_epochs= 11, minibatch_size = 100, learning_rate = 0.04, save= False)

#Load model
tf.reset_default_graph()  #jut for test

_, Nv= X_tr.shape #m= #of users,  number of visible units Nv = number of movies
N_hidden =10
rating_scale=2

reco= rbm.RBM(Nv, N_hidden, rating_scale) #Initialize the RBM.

saver = tf.train.Saver()

with tf.Session() as sess:

    reco.set_session(sess)
    saved_files = saver.restore(sess, 'saver/rbm_model_saver.ckpt')
    out_v, out_pvh = reco.predict(X_tr)


#Create a user report with recommendations
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
