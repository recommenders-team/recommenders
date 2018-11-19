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
import tensorflow as tf
from scipy import sparse #to create the rating matrix
import logging

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    PREDICTION_COL,
)

log = logging.getLogger(__name__)

class RBM(object):

    #initialize class parameters
    def __init__(
        self,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        hidden_units,
        keep_prob,
        init_stdv = 0.01,
        learning_rate= 0.004,
        minibatch_size= 100,
        training_epoch
        save = False,
        save_path = 'saver/rbm_model_saver.ckpt',
        debug = False,
    ):

        #pandas DF parameters
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user

        #RBM parameters
        self.Nh_ = hidden_units    #number of hidden units
        self.keep = keep_prob       #keep probability for dropout regularization
        self.std = init_stdv          #standard deviation used to initialize the weights matrices
        self.alpha = learning_rate   #learning rate used in the update method of the optimizer

        #size of the minibatch used in the random minibatches training. This should be set
        #approx between  1 - 120. setting to 1 correspods to stochastic gradient descent, and
        #it is considerably slower.Good performance is achieved for a size of ~100.
        self.minibatch= minibatch_size
        self.epochs= training_epoch  #number of epochs used to train the model
        self.save = save #If true, it saves a TF model to be used for predictions
        self.save_path = save_path #specify a path where the TF model file is saved


    #===============================================
    #Generate the Ranking matrix from a pandas DF
    #===============================================

    def gen_ranking_matrix(df):

        '''
        Generate the user/item rating matrix using scipy's sparse matrix method coo_matrix;
        for reference see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

        The input format is coo_matrix((data, (rows, columns)), shape=(rows, columns))

        Basic mechanics: take the maximum index of both userID and ItemID and generates a
        [1,max(UserID)]x[1, max(ItemID)] matrix. If the original dataset has no missing values
        either in UserID or ItemID, the resulting matrix will not contain any rows or columns of
        all zeroes.

        Example 1: a user is missing, then the matrix RM[missing_user, :] = [0,0,0,...,0]. This
                    may hurts the perfomance of the rbm, more testing necessary
        Example 2: No user has rated a particular movie, RM[:, missing_item] = [0,0,0,...,0]. The
                   rbm will still tries to generate ratings for this movie using the ratings given
                   by the users on similar movies.

        Args:
            df: A dataframe containing at least UserID, ItemID, Ratings

        Returns:
            RM: user-affinity matrix of dimensions (Nusers, Nitems) in numpy format. Unrated movies
            are assigned a value of 0.

        '''
        log.info("Collecting user affinity matrix...")

        rating = df.sort_values(by=[self.col_user])

        #find max user and item index
        Nusers = rating[self.col_user].max()
        Nitems = rating[self.col_item].max()

        #extract informations from the dataframe as an array. Note that we substract 1 from itm_id and usr_id
        #in order to map it to matrix format

        r_ = rating[self.col_rating].values #ratings
        itm_id =(rating[self.col_item]-1).values #itm_id serving as columns
        usr_id =(rating[self.col_user]-1).values  #usr_id serving as rows

        #check that all 3 vectors have the same dimensions
        assert((usr_id.shape[0]== r_.shape[0]) & (itm_id.shape[0] == r_.shape[0]))

        #generate a sparse matrix representation using scipy's coo_matrix and convert to array format
        RM = coo_matrix((r_, (usr_id, itm_id)), shape= (Nusers, Nitems)).toarray()

        return RM


    #==================================
    #Define graph topology
    #==================================

    #Initialize the placeholders for the visible units
    def placeholder(self):
        self.v = tf.placeholder(shape= [None, self.Nv_], dtype= 'float32')
        return self.v

    #initialize the parameters of the model.
    def init_parameters(self):

        '''
        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and two bias vector to initialize.

        Arguments:
        Nv -- number of visible units (input layer)
        Nh -- number of hidden units (latent variables of the model)

        Returns:
        Initialized weights and biases. We initialize the transition matrix by sampling from a normal distribution with zero mean
        and given variance. The biases are Initialized to zero.
        '''

        tf.set_random_seed(1) #set the seed for the random number generator

        with tf.variable_scope('Network_parameters'):

            self.w  = tf.get_variable('weight',  [self.Nv_, self.Nh_], initializer = tf.random_normal_initializer(stddev=self.std, seed=1), dtype= 'float32' )
            self.bv = tf.get_variable('v_bias',  [1, self.Nv_], initializer= tf.zeros_initializer(), dtype= 'float32' )
            self.bh = tf.get_variable('h_bias',  [1, self.Nh_], initializer= tf.zeros_initializer(), dtype='float32')

    #=========================
    #Helper functions
    #========================

    def set_session(self, session):
        self.session = session

    def B_sampling(self,p):
        '''
        Sample from a Binomial distribution.

        1) Extract a random number from a uniform distribution (U) and compare it with the unit's probability (P)
        2) Choose 0 if P<U, 1 otherwise

        P -- input conditional probability
        U -- Bernoulli probability used for comparison

        sampled -- sampled units. The value is 1 if P>U and 0 otherwise. It is convenient to implement
                   this condtivk = reco.G_sampling(k= 1)on using the relu function.

        '''
        np.random.seed(1)

        #sample from a Bernoulli distribution with same dimensions as input distribution
        g = np.random.uniform(size=p.shape[1] )

        #sample the
        h_sampled = tf.nn.relu(tf.sign(p-g) )

        return h_sampled


    def M_sampling(self, p):

        '''
        Multinomial Sampling

        For K classes, we sample K binomial distributions using the acceptance/Rejection method. This is possible
        since each class is statistically independent form the otherself. Note that this is the same method used
        in Numpy's random.multinomial() function.

        Using broadcasting along the 3rd index, this function can be easily implemented

        Args:
        p -- a distributions of shape (m, n, r), where m is the number of examples, n the number of features and
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


    def Pm(self, phi):

        '''
        Probability that unit v has value l given phi: P(v=l|phi)

        Argument:
        phi -- linear combination of values of the previous layer
        r   -- rating scale, corresponding to the number of classes

        Returns
        pk -- a tensor of shape (r, m, Nv) . This needs to be reshaped as (m, Nv, r) in the last step
        to allow for faster sampling when used in the Multinomial function.

        '''

        num = [tf.exp(tf.multiply(tf.constant(k, dtype='float32'),phi)) for k in range(1,self.r_+1)]
        den = 1+tf.reduce_sum(num, axis=0)

        pk = tf.div(num, den)

        return tf.transpose(pk, perm= [1,2,0])


    def Fv(self, x):

        '''
        Free energy of the visible units given the hidden units. Since the sum is over the hidden units' states, the
        functional form of the visible units Free energy is the same as the one for the binary model.

        Arguments:
        x  -- This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
        F -- Free energy of the model
        '''

        b = tf.reshape(self.bv, (self.Nv_, 1))
        bias = -tf.matmul(x, b)
        bias = tf.reshape(bias, (-1,))

        phi_x = tf.matmul(x, self.w)+ self.bh
        f = - tf.reduce_sum(tf.nn.softplus(phi_x))

        F  = bias + f #free energy density per training example

        return F

    #===================
    #Sampling
    #===================

    '''
    Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
    to initialize the two conditional probabilities:

    P(h=1|phi_v) --> returns the probability that the i-th hidden unit is active
    P(v=l|phi_h) --> returns the probability that the  i-th visible unit is in state l !=0

    '''

    #sample the hidden units
    def sample_h(self, vv):
        '''
        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Arguments:
        vv -- visible units tensor

        Returns:
        phv -- activation probability of the hidden unit
        h_  -- sampled value of the hidden unit from a Bernoulli distributions having success probability phv

        Note: we use Bernoulli sampling from TF probability module. It is essential to specify the dtype of the
        event samples when initializing the distribution. In this case, as phv is float32, dtype = float32 as
        well.

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

        Arguments:
        h -- visible units tensor

        Returns:
        pvh -- activation probability of the visible unit given the hidden
        v_  -- sampled value of the visible unit from a Multinomial distributions having success probability pvh. There are two
               steps here:

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
        impose that unseen movies stay as such, i.e. the sampling phase should not modify the elelments where v =-1.

        Arguments:
        k -- iterator. Number of sampling steps
        v -- visible units

        Returns:
        h_k -- sampled value of the hidden unit at step  k
        v_k -- sampled value of the visible unit at step k

        '''
        v_k = self.v #initialize the value of the visible units at step k=0 on the data

        for i in range(k): #k_sampling
            _, h_k = self.sample_h(v_k)
            _ ,v_k = self.sample_v(h_k)

        return v_k

    #2 Contrastive divergence
    def Losses(self, vv, v_k):

        '''
        Loss functions

        Arguments:
        v     -- empirical input
        v_k   -- sampled visible units at step k

        Returns:
        obj  -- objective function of Contrastive divergence, that is the difference between the
                free energy clamped on the data (v) and the model Free energy (v_k)

        '''

        with tf.variable_scope('losses'):

            obj  = tf.reduce_mean(self.Fv(vv) - self.Fv(v_k))

        return obj

    #================================================
    # model performance (online metrics)
    #================================================

    #inference
    def infere(self):
        '''
        Prediction: A training example is used to activate the hidden unit that in turns produce new ratings for the
                visible units, both for the rated and unrated examples.

        Argument:
        xtr -- example from dataset. This can be either the test/train set

        Returns:
        pred -- inferred values

        '''

        #predict a new value
        _, h_p = self.sample_h(self.v)
        pvh ,vp = self.sample_v(h_p)

        return pvh, vp

    #Metrics
    def accuracy(self, vp):

        '''
        Train/Test Classification Accuracy

        Evaluates the accuracy over the train/test set in online mode. Note that this needs to be evaluated on the rated
        items only

        Arguments
        vp -- infereed output (Network prediction)

        Returns
        ac_score =  1/m Sum_{mu, i} I(vp-test = 0)_{mu,i}

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

    #instantiate the computational graph
    self.placeholder()
    self.init_parameters()

    def train(self, xtr, xtst, epochs, alpha, minibatch_size):

        '''
        Training ops

        '''
        m, _ = xtr.shape #dimension of the input: #examples, #features
        num_minibatches = int(m / minibatch_size)

        #-------------------Initialize all parameters----------------
        #Lists to collect the metrics across each epochs
        Mse_train = []

        k=1 #initialize the G_sampling step
        epoch_sample = [50, 70, 80,90] #learning percentage to increment the k-step
        l=0

        v_k = self.G_sampling(k)

        obj = self.Losses(self.v, v_k) #objective function
        rate = alpha/minibatch_size #rescaled learning rate

        opt = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(loss= obj) #optimizer

        pvh, vp = self.infere() #sample the value of the visible units given the hidden. Also returns and the related probabilities

        #initialize metrics
        Mserr  = self.msr_error(v_k)
        Clacc  = self.accuracy(v_k)

        #start loop over training epochs
        for i in range(epochs):

            epoch_tr_err =0
            per= (i/epochs)*100 #percentage of epochs

            #Increase the G_sampling step k at each learning percentage specified in the epoch_sample vector (to improve)
            if per!=0 and per %epoch_sample[l] == 0:
                k +=1
                l +=1
                v_k = self.G_sampling(k)

            #implement minibatches (try using TF data pipeline for performance)
            minibatches = random_mini_batches(xtr, mini_batch_size= minibatch_size, seed=1)

            for minibatch in minibatches:

                _, batch_err = self.session.run([opt, Mserr], feed_dict={self.v:minibatch})

                epoch_tr_err += batch_err/num_minibatches #mse error per minibatch

            if i%5==0:
                print('training epoch %i rmse Train %f ' %(i, epoch_tr_err) )

            #write metrics acros epohcs
            Mse_train.append(epoch_tr_err) # mse training error per training epoch

        #Final Classification metrics over the entire dataset

        #MSRE
        Mse_tr = self.session.run(Mserr, feed_dict={self.v:xtr})
        Mse_test  = self.session.run(Mserr, feed_dict={self.v:xtst})

        print('MSR error: Train  %f, Test: %f'%(Mse_tr, Mse_test) )

        #Classification
        train_acc = self.session.run(Clacc, feed_dict={self.v:xtr})
        test_acc = self.session.run(Clacc, feed_dict={self.v:xtst})

        print('Classification Accuracy: Train  %f, Test: %f'%(train_acc, test_acc) )

        return Mse_train, Mse_test

    #=========================
    # load a pretrained model
    #=========================

    #Inference from a trained model. This can be either loaded from a saved model or in the same sessions as the traiing one
    def predict(self, x):

        '''
        Prediction: A training example is used to activate the hidden unit that in turns produce new ratings for the
        visible units, both for the rated and unrated examples.

        Argument:
        x -- example from dataset

        Returns:
        pred -- inferred values

        '''
        #Sampling
        _, h_ = self.sample_h(self.v) #sample h

        #sample v
        phi_h  = tf.matmul(h_, tf.transpose(self.w))+ self.bv #linear combination
        pvh = self.Pm(phi_h) #conditional probability of v given h

        v_  = self.M_sampling(pvh) #sample the value of the visible units

        #evaluate v on the data
        vp, p = self.session.run([v_, pvh], feed_dict={self.v: x})

        return vp, p
