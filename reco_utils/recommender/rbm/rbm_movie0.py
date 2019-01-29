'''

Movie recommender with multinomial RBM v.0.2 (2018)
============================================================
Author: Mirco Milletari <mirco.milletari@microsoft.com>


'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from reco_utils.recommender.rbm.rbm import RBM
from reco_utils.dataset.numpy_splitters import numpy_stratified_split
from reco_utils.dataset.sparse import AffinityMatrix

#For interactive mode only
%load_ext autoreload
%autoreload 2

from reco_utils.dataset import movielens


MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['userID','movieID','rating','timestamp']
)

# Convert to 32-bit in order to reduce memory consumption
data.loc[:, 'rating'] = data['rating'].astype(np.int32)

data.head()


#to use standard names across the analysis
header = {
        "col_user": "userID",
        "col_item": "movieID",
        "col_rating": "rating",
    }

#instantiate the sparse matrix generation
am = AffinityMatrix(DF = data, **header)

#obtain the sparse matrix
X = am.gen_affinity_matrix()
Xtr, Xtst = numpy_stratified_split(X)


#First we initialize the model class
model = RBM(hidden_units= 600, training_epoch = 20, minibatch_size= 60, keep_prob=0.9,with_metrics = False)
train_time= model.fit(Xtr, Xtst)
