# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

'''
Implementation of a multinomial Restricted Boltzmann Machine for collaborative filtering
in numpy/pandas/tensorflow

Based on the article by Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton
https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

In this implementation we use multinomial units instead of the one-hot-encoded used in
the paper.This means that the weights are rank 2 (matrices) instead of rank 3 tensors.

Basic mechanics:

1) A computational graph is created when the RBM class is instantiated;
        For an items based recommender this consists of:
        -- visible units: The number Nv of visible units equals the number of items
        -- hidden units : hyperparameter to fix during training

2) Sampling via Contrastive Divergence (Bernoulli sampling)
        2.1) for each training epoch, the visible units are first clamped on the data
        2.2) The activation probability of the hidden units, given a linear combination of
             the visibles, is evaluated P(h=1|phi_v). The latter is then used to sample the
             value of the hidden units.
        2.3) The probability P(v=l|phi_h) is evaluated, where l=1,..,r are the rates (e.g.
             r=5 for the movielens dataset). In general, this is a multinomial distribution,
             from which we sample the value of v.
        2.4) This step is repeated k times, where k increases as optimization converges. It is
             essential to fix to zero the original unrated items during the all learning process.

3) Optimization
         The free energy of the visible units given the hidden is evaluated at the beginning (F_0)
          and after k steps of Bernoulli sampling (F_k). The weights and biases are updated by
          minimizing the differene F_0 - F_k.

4) Inference
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users

'''

#import libraries
import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt

import tensorflow as tf
import logging

#import default parameters
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    PREDICTION_COL,
)

#import default parameters defined in the local init file
from reco_utils.recommender.rbm import (
    HIDDEN,
    KEEP_PROB,
    STDV,
    ALPHA,
    MINIBATCH,
    EPOCHS,
    MOMENTUM,
    DEFAULTPATH,
    _user_item_return_type,
    _predict_column_type,
)

#for logging
log = logging.getLogger(__name__)

class RBM:

    #initialize class parameters
    def __init__(
        self,
        remove_seen=True,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        hidden_units= HIDDEN,
        keep_prob= KEEP_PROB,
        momentum = MOMENTUM,
        init_stdv = STDV,
        learning_rate= ALPHA,
        minibatch_size= MINIBATCH,
        training_epoch= EPOCHS,
        save_model= False,
        save_path = 'reco_utils/recommender/rbm/saver',
        debug = False,
    ):

        self.remove_seen = remove_seen #if True it removes from predictions elements in the train set

        #pandas DF parameters
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user

        #RBM parameters
        self.Nh_ = hidden_units     #number of hidden units
        self.keep = keep_prob       #keep probability for dropout regularization
        self.momentum_= momentum    #initial value of the momentum for the optimizer
        self.std = init_stdv        #standard deviation used to initialize the weights matrices
        self.alpha = learning_rate  #learning rate used in the update method of the optimizer

        #size of the minibatch used in the random minibatches training. This should be set
        #approx between  1 - 120. setting to 1 correspods to stochastic gradient descent, and
        #it is considerably slower.Good performance is achieved for a size of ~100.
        self.minibatch= minibatch_size
        self.epochs= training_epoch  #number of epochs used to train the model

        #Options to save the model for future use
        self.save_model_ = save_model
        self.save_path_ = save_path


    #=========================
    #Helper functions
    #========================

    #Binomial sampling
    def B_sampling(self,p):

        '''
        Sample from a Binomial distribution.

        1) Extract a random number from a uniform distribution (U) and compare it with the unit's probability (P)
        2) Choose 0 if P<U, 1 otherwise

        Args:
            P: input conditional probability
            U: Bernoulli probability used for comparison

        Returns:
            h_samples: sampled units. The value is 1 if P>U and 0 otherwise. It is convenient to implement
                       this condtivk = reco.G_sampling(k= 1)on using the relu function.

        '''
        np.random.seed(1)

        #sample from a Bernoulli distribution with same dimensions as input distribution
        g = np.random.uniform(size=p.shape[1] )

        #sample the
        h_sampled = tf.nn.relu(tf.sign(p-g) )

        return h_sampled

    #Multinomial sampling
    def M_sampling(self, p):

        '''
        Multinomial Sampling

        For r classes, we sample r binomial distributions using the acceptance/Rejection method. This is possible
        since each class is statistically independent form the otherself. Note that this is the same method used
        in Numpy's random.multinomial() function.

        Using broadcasting along the 3rd index, this function can be easily implemented

        Args:
            p:  a distributions of shape (m, n, r), where m is the number of examples, n the number of features and
                r the number of classes. p needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.

        Returns:
            v_samp: an (m,n) array of sampled values from 1 to r . Given the sampled distribution of the type [0,1,0, ..., 0]
            it returns the index of the value 1 . The outcome is index = argmax() + 1 to account for the fact that array indices
            start from 0 .

        '''
        np.random.seed(1)

        g = np.random.uniform(size=p.shape[2] )

        samp = tf.nn.relu(tf.sign(p - g) )

        v_samp = tf.cast( tf.argmax(samp, axis= 2)+1, 'float32')

        return v_samp

    #Multinomial distribution
    def Pm(self, phi):

        '''
        Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi: linear combination of values of the previous layer
            r  : rating scale, corresponding to the number of classes

        Returns:
            pr: a tensor of shape (r, m, Nv) . This needs to be reshaped as (m, Nv, r) in the last step
                to allow for faster sampling when used in the Multinomial function.

        '''

        num = [tf.exp(tf.multiply(tf.constant(k, dtype='float32'),phi)) for k in range(1,self.r_+1)]
        den = tf.reduce_sum(num, axis=0)

        pr = tf.div(num, den)

        return tf.transpose(pr, perm= [1,2,0])

    #Free energy
    def Fv(self, x):

        '''
        Free energy of the visible units given the hidden units. Since the sum is over the hidden units' states, the
        functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x: This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            F: Free energy of the model
        '''

        b = tf.reshape(self.bv, (self.Nv_, 1))
        bias = -tf.matmul(x, b)
        bias = tf.reshape(bias, (-1,))

        phi_x = tf.matmul(x, self.w)+ self.bh
        f = - tf.reduce_sum(tf.nn.softplus(phi_x))

        F  = bias + f #free energy density per training example

        return F

    #Random minibatches function
    def random_mini_batches(self, X, seed = 1):

        """
        Creates a list of random minibatches from X

        Args:
            X: input data, of shape (input size, number of examples) (m, ne)
            self.minibatch: size of the mini-batches (integer)
            seed: random seed

        Returns:
            mini_batches: list
        """

        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]

        # Step 2: Partition  Minus the end case.
        num_complete_minibatches = math.floor(m/self.minibatch) # number of mini batches of size mini_batch_size

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * self.minibatch : k * self.minibatch + self.minibatch]
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.minibatch != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * self.minibatch : m]
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)

        return mini_batches

    #==================================
    #Define graph topology
    #==================================

    #Initialize graph
    #with self.graph.as_default():

    #Initialize the placeholders for the visible units
    def placeholder(self):
        self.v = tf.placeholder(shape= [None, self.Nv_], dtype= 'float32')

    #initialize the parameters of the model.
    def init_parameters(self):

        '''
        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and
        two bias vector to initialize.

        Arguments:
        Nv -- number of visible units (input layer)
        Nh -- number of hidden units (latent variables of the model)

        Returns:
        Initialized weights and biases. We initialize the transition matrix by sampling from a
        normal distribution with zero mean and given variance. The biases are Initialized to zero.
        '''

        tf.set_random_seed(1) #set the seed for the random number generator

        with tf.variable_scope('Network_parameters'):

            self.w  = tf.get_variable('weight',  [self.Nv_, self.Nh_], initializer = tf.random_normal_initializer(stddev=self.std, seed=1), dtype= 'float32' )
            self.bv = tf.get_variable('v_bias',  [1, self.Nv_], initializer= tf.zeros_initializer(), dtype= 'float32' )
            self.bh = tf.get_variable('h_bias',  [1, self.Nh_], initializer= tf.zeros_initializer(), dtype='float32')


    #===================
    #Sampling
    #===================

    '''
    Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
    to initialize the two conditional probabilities:

    P(h|phi_v) --> returns the probability that the i-th hidden unit is active
    P(v|phi_h) --> returns the probability that the  i-th visible unit is active

    '''

    #sample the hidden units
    def sample_h(self, vv):

        '''
        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv: visible units tensor

        Returns:
            phv: activation probability of the hidden unit
            h_ : sampled value of the hidden unit from a Bernoulli distributions having success probability phv

        '''

        with tf.name_scope('sample_hidden_units'):

            phi_v = tf.matmul(vv, self.w)+ self.bh #create a linear combination
            phv   = tf.nn.sigmoid(phi_v) #conditional probability of h given v
            phv_reg= tf.nn.dropout(phv, self.keep)

            #Sampling
            h_  = self.B_sampling(phv_reg) #obtain the value of the hidden units via Bernoulli sampling

        return phv, h_

    #sample the visible units
    def sample_v(self, h):

        '''
        Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN (negative phase)
        Each visible unit can take values in [1,r], while the zero is reserved for missing data; as such the value of the
        hidden unit is sampled from a multinomial distribution.

        Args:
            h: visible units tensor

        Returns:
            pvh: activation probability of the visible unit given the hidden
            v_ : sampled value of the visible unit from a Multinomial distributions having success probability pvh. There are two
                 steps here:

        Basic mechanics:
           1) For every training example we first sample Nv Multinomial distributions. The result is of the form [0,1,0,0,0,...,0]
              where the index of the 1 element corresponds to the rth rating. The index is extracted using the argmax function and
              we need to add 1 at the end since array indeces starts from 0.

           2) Selects only those units that have been sampled. During the training phase it is important to not use the reconstructed
              inputs, so we beed to enforce a zero value in the reconstructed ratings in the same position as the original input.

        '''

        with tf.name_scope('sample_visible_units'):

            phi_h  = tf.matmul(h, tf.transpose(self.w))+ self.bv #linear combination
            pvh = self.Pm(phi_h) #conditional probability of v given h

            #Sampling (modify here )
            v_tmp  = self.M_sampling(pvh) #sample the value of the visible units

            mask = tf.equal(self.v, 0) #selects the inactive units in the input vector
            v_ = tf.where(mask, x = self.v, y = v_tmp) #enforce inactive units in the reconstructed vector

        return pvh, v_

    #=======================
    #Training ops
    #=======================
    '''
    Training in generative models takes place in two steps:

    1) Gibbs sampling
    2) Gradient evaluation and parameters update

    This estimate is later used in the weight update step by minimizing the distance between the
    model and the empirical free energy. Note that while the unit's configuration space is sampled,
    the weights are determined via maximum likelihood (saddle point).
    '''

    #1) Gibbs Sampling

    def G_sampling(self,k):

        '''
        Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary RBM we need to
        impose that unseen movies stay as such, i.e. the sampling phase should not modify the elelments where v =0.

        Args:
            k: iterator. Number of sampling steps
            v: visible units

        Returns:
            h_k: sampled value of the hidden unit at step  k
            v_k: sampled value of the visible unit at step k

        '''
        v_k = self.v #initialize the value of the visible units at step k=0 on the data

        for i in range(k): #k_sampling
            _, h_k = self.sample_h(v_k)
            _ ,v_k = self.sample_v(h_k)

        return v_k

    #2) Contrastive divergence
    def Losses(self, vv, v_k):

        '''
        Loss functions

        Args:
            v: empirical input
            v_k: sampled visible units at step k

        Returns:
            obj: objective function of Contrastive divergence, that is the difference between the
                 free energy clamped on the data (v) and the model Free energy (v_k)

        '''

        with tf.variable_scope('losses'):

            obj  = tf.reduce_mean(self.Fv(vv) - self.Fv(v_k))

        return obj

    #================================================
    # model performance (online metrics)
    #================================================

    #Inference
    def infere(self):

        '''
        Prediction: A training example is used to activate the hidden unit that in turns produce new ratings for the
                        visible units, both for the rated and unrated examples.

        Args:
            xtr: example from dataset. This can be either the test/train set

        Returns:
            pred: inferred values

        '''

        #predict a new value
        _, h_p = self.sample_h(self.v)
        pvh ,vp = self.sample_v(h_p)

        return pvh, vp

    #Metrics
    def map(self, vp):

        '''
        Train/Test Mean average precision

        Evaluates MAP over the train/test set in online mode. Note that this needs to be evaluated on the rated
        items only

        Args:
            vp: inferred output (Network prediction)

        Returns:
            map_score =  1/m Sum_{mu=1}^{m} Sum{i=1}^Nv 1/s(i) I(v-vp = 0)_{mu,i}
            where m = Nusers, Nv = number of items = number of visible units and s(i) is the number of non-zero
            elements per row.

        '''
        with tf.name_scope('accuracy'):

            #1) define and apply the mask
            mask= tf.not_equal(self.v,0)
            n_values= tf.reduce_sum(tf.cast(mask, 'float32'), axis=1)

            #2) Take the difference between the input data and the inferred ones. This value is zero whenever the two
            #   values coincides
            vd = tf.where(mask, x= tf.abs(tf.subtract(self.v,vp)), y= tf.ones_like(self.v) )

            #correct values: find the location where v = vp
            corr = tf.cast(tf.equal(vd, 0), 'float32' )

            #3) evaluate the accuracy
            ac_score = tf.reduce_mean(tf.div(tf.reduce_sum(corr, axis= 1), n_values) )

        return ac_score


    def msr_error(self, vp):

        '''
        Mean square root error

        Note that this needs to be evaluated on the rated items only

        '''

        with tf.name_scope('msr_error'):

            mask= tf.not_equal(self.v,0) #selects only the rated items
            n_values= tf.reduce_sum(tf.cast(mask, 'float32'), axis=1) #number of rated items

            #evaluate the square difference between the inferred and the input data on the rated items
            e= tf.where(mask, x= tf.squared_difference(self.v, vp), y= tf.zeros_like(self.v) )

            #evaluate the msre
            err = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_sum(e, axis= 1), n_values))/2)

        return err


    #=========================
    # Training ops
    #=========================

    def fit(self, xtr, xtst):

        '''
        Fit method

        Main component of the algo; once instantiate, it generates the computational graph

        Args:
            Xtr: the user/affinity matrix for the train set

        Returns:
            trained_param: the trained values for the weight matrix and the visible/hidden bias vectors
            msre : the mean square root error per training epoch
            precision: the mean average precision over the entire dataset

        Baic mechanics:

        '''

        self.r_= xtr.max() #defines the rating scale, e.g. 1 to 5
        m, self.Nv_ = xtr.shape #dimension of the input: m= N_users, Nv= N_items

        num_minibatches = int(m / self.minibatch) #number of minibatches
        self.epochs = self.epochs +1 #add one epoch

        tf.reset_default_graph()
        #----------------------Initialize all parameters----------------

        log.info("Creating the computational graph")
        #instantiate the computational graph
        self.placeholder()
        self.init_parameters()

        #--------------Sampling protocol for Gibbs sampling-----------------------------------
        k=1 #initialize the G_sampling step
        l=0 #initialize epoch_sample index
        #Percentage of the total number of training epochs after which the k-step is increased
        epoch_sample = [50, 70, 80,90]

        #-------------------------Main algo---------------------------

        v_k = self.G_sampling(k) #sampled value of the visible units

        obj = self.Losses(self.v, v_k) #objective function
        rate = self.alpha/self.minibatch  #rescaled learning rate

        opt = tf.contrib.optimizer_v2.MomentumOptimizer(learning_rate = rate, momentum = self.momentum_).minimize(loss= obj) #optimizer

        pvh, vp = self.infere() #sample the value of the visible units given the hidden. Also returns  the related probabilities

        #initialize online metrics
        Mse_train = [] #Lists to collect the metrics across each epochs

        #Metrics
        Mserr  = self.msr_error(v_k)
        #Clacc  = self.map(v_k)

        if self.save_model_:
            saver = tf.train.Saver() #save the model to file

        init_g = tf.global_variables_initializer() #Initialize all variables

        #Start TF session on default graph
        self.sess = tf.Session()

        self.sess.run(init_g)

        #start loop over training epochs
        for i in range(self.epochs):

            epoch_tr_err =0 #initialize the training error for each epoch to zero
            per= (i/self.epochs)*100 #current percentage of the total #epochs

            #Increase the G_sampling step k at each learning percentage specified in the epoch_sample vector (to improve)
            if per !=0 and per %epoch_sample[l] == 0:
                k +=1
                l +=1
                v_k = self.G_sampling(k)

            #minibatches (to implement: TF data pipeline for better performance)
            minibatches = self.random_mini_batches(xtr)

            for minib in minibatches:

                _, batch_err = self.sess.run([opt, Mserr], feed_dict={self.v:minib})

                epoch_tr_err += batch_err/num_minibatches #average mse error per minibatch

            if i % 10==0:
                print('training epoch %i rmse Train %f ' %(i, epoch_tr_err) )

            #write metrics acros epohcs
            Mse_train.append(epoch_tr_err) # mse training error per training epoch

            #precision_train, self.trained_param = sess.run([Clacc, tf.trainable_variables()], feed_dict={self.v: xtr})
            #precision_test = sess.run(Clacc, feed_dict={self.v:xtst})

        if self.save_model_:
            saver.save(sess, self.save_path_ + '/rbm_model_saver.ckpt')

        #Print training error as a function of epochs
        plt.plot(Mse_train, label= 'train')
        plt.ylabel('msr_error', size= 'x-large')
        plt.xlabel('epochs', size = 'x-large')
        plt.legend(ncol=1)

        #Final precision scores
        #print('MAP on the train set', precision_train)
        #print('Total precision on the test set', precision_test)
        #print('train/test difference', precision_train - precision_test)

        return self.sess

    #=========================
    # Inference modules
    #=========================

    def eval_out(self,x):

        '''
        Implement multinomial sampling from a trained model

        Args:


        '''
        #Sampling
        _, h_ = self.sample_h(self.v) #sample h

        #sample v
        phi_h  = tf.matmul(h_, tf.transpose(self.w))+ self.bv #linear combination
        pvh_ = self.Pm(phi_h) #conditional probability of v given h

        v_  = self.M_sampling(pvh_) #sample the value of the visible units

        return v_, pvh_


    def recommend_k_items(self, x, maps, top_k=10):

        '''
        Returns the top-k items ordered by a relevancy score

        Args:
            x: input user/affinity matrix. Note that this can be a single vector, i.e. the ratings of a single user.
            top_k: the number of items to recommend
            maps : dictionaries used to associate the original item/user IDs when constructing the pd dataframe

        Returns:
            results: a pd dataframe containing the top_k elements togetehr with their score

        Basic mechanics:
            The method can be called either within the same session or by restoring a previous session from file.
            If save_model is true, a graph is generated and then populated with the pre trained values of the
            parameters. Otherwise, the default session used during training is usedself.

            The method samples new ratings from the learned joint distribution, together with their probabilities.
            The input x must have the same number of columns of the one used for training the model, i.e. the same
            number of items, but it can have an arbitrary number of rows (users).

            A recommendation score is evaluated by taking the element-wise product between the rating and the
            associated probability. For example, we could have the following situation:

                    rating     probaility     score
            item1     5           0.5          2.5
            item2     4           0.8          3.2

            then item2 will be recommended. Finally, a dataframe is generated containing the first top_k items,
            in decreasing score order. A map_back dictionary is used to relate the recommended matrix elements
            to their IDs in the original dataframe.

        '''

        if self.save_model_: #if true, restore the computational graph from a trained session

            m, self.Nv_ = x.shape #dimension of the input: m= N_users, Nv= N_items

            self.r_= x.max() #defines the rating scale, e.g. 1 to 5

            tf.reset_default_graph()

            self.placeholder()
            self.init_parameters()

            saver = tf.train.Saver()

            self.sess = tf.Session()
            saved_files = saver.restore(self.sess,  self.save_path_ + '/rbm_model_saver.ckpt')

        else: m, _ = x.shape #dimension of the input: m= N_users, Nv= N_items

        v_, pvh_ = self.eval_out(x) #evaluate the ratings and the associated probabilities

        #evaluate v_ and pvh_ on the input data
        vp, pvh= self.sess.run([v_, pvh_], feed_dict={self.v: x})

        pv= np.max(pvh, axis= 2) #returns only the probabilities for the predicted ratings in vp

        #evaluate the score
        score =  np.multiply(vp, pv)

        #----------------------Return the results as a P dataframe------------------------------------

        top_items  = np.argpartition(-score, range(top_k), axis= 1)[:,:top_k] #get the top k items
        top_scores = score[np.arange(score.shape[0])[:, None], top_items] #get top k scores

        top_items_ = np.reshape(np.array(top_items), -1)
        top_scores_ = np.reshape(np.array(top_scores), -1)

        #generates userids
        userids = []
        for i in range(0, m):
            userids.extend([i]*top_k)

        #create dataframe
        results = pd.DataFrame.from_dict(
            {
                self.col_user: userids,
                self.col_item: top_items_,
                self.col_rating: top_scores_,
            }
        )

        map_back_users = maps[0]
        map_back_items = maps[1]

        # remap user and item indices to IDs
        results[self.col_user] = results[self.col_user].map(map_back_users)
        results[self.col_item] = results[self.col_item].map(map_back_items)

        # format the dataframe in the end to conform to Suprise return type
        log.info("Formatting output")

        # reformatting the dataset for the output
        return (
            results[[self.col_user, self.col_item, self.col_rating]]
            .rename(columns={self.col_rating: PREDICTION_COL})
            .astype(
                {
                    self.col_user: _user_item_return_type(),
                    self.col_item: _user_item_return_type(),
                    PREDICTION_COL: _predict_column_type(),
                }
            )
        )
