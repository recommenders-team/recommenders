'''

Movie recommender with multinomial RBM v.0.2 (2018)
============================================================
Author: Mirco Milletari <mirco.milletari@microsoft.com>

Restricted Boltzmann machines are used to perform collaborative filtering over the Movielens dataset.
The model generates new ratings given the collected ratings. Model performance is also analysed over
different choices of the hyper parameter.

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import time as tm

from helperfunct import*

#ML Libraries and methods
import tensorflow as tf
import MNrbm as rbm

#For interactive mode only
%load_ext autoreload
%autoreload 2


#Load the movielens dataset
dir = os.getcwd() #optain the absolute path
path1 = dir + '/movielens_data/movies.dat'

movies_df = pd.read_csv(path1, sep='::', header= None, engine= 'python' ) #movies dataset
movies_df.columns = ['ItemID', 'Title', 'Genre']

#inspect first entries
movies_df.head()


#Number of unique genres
gen = ['Action','Adventure', 'Animation','Children', 'Comedy','Crime','Documentary','Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

len(gen)

#load the ratings
path2 = dir+ '/movielens_data/ratings.dat'
ratings_df = pd.read_csv(path2, sep='::', header=None, engine = 'python')

ratings_df.columns = ['UserID', 'ItemID', 'Rating', 'Time']

ratings_df.head()




#==========================
#Data processing
#==========================
#generate the ranking matrix. Movies that haven not been rated yet get a 0 rating.

#generate using scipy method
#st1= tm.time()
RX = gen_ranking_matrix(ratings_df)
#ed1= tm.time()

#elapsed time
#ed1-st1

#generate using the old method with a for loop
#st2 = tm.time()
#RM = gen_ranking_matrix_v0(ratings_df)
#ed2= tm.time()

#ed2-st2

#----------------------------------------------------
#Need to check the influenc of including the unrated movies
#this is easly done by using unique. For movielens, there are 256 movies that
#have not been rated by anyone. Exclude them and compare the result with the inclusion version 


unique_users = ratings_df['UserID'].unique()
unique_items = ratings_df['ItemID'].unique()

unique_users.shape
unique_items.shape

RX.shape






#Distribution of values
zeros  = (RX == 0).sum()
total  = RX.shape[0]*RX.shape[1]

id =np.where(RX !=0)

plt.hist(RX[id], bins=5, density=1)

#Percentage of zeros in the matrix
zeros/total *100

#cut a % for the test set
X_tr, X_tst = train_test_split(RX, test_size= 20)

#Binary rating (optional)
B_tr = binary_rating(X_tr, 2)
B_tst= binary_rating(X_tst,2)

ib =np.where(RB !=0)

plt.hist(RB[ib], bins=2, density=1)

#RS = rescale(RX, 5,3)
#np.save('Xtrain', X_tr)

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
