# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
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

2) Gibbs Sampling:
        2.1) for each training epoch, the visible units are first clamped on the data
        2.2) The activation probability of the hidden units, given a linear combination of
             the visibles, is evaluated P(h=1|phi_v). The latter is then used to sample the
             value of the hidden units.
        2.3) The probability P(v=l|phi_h) is evaluated, where l=1,..,r are the rates (e.g.
             r=5 for the movielens dataset). In general, this is a multinomial distribution,
             from which we sample the value of v.
        2.4) This step is repeated k times, where k increases as optimization converges. It is
             essential to fix to zero the original unrated items during the all learning process.

3) Optimization:
         The free energy of the visible units given the hidden is evaluated at the beginning (F_0)
         and after k steps of Bernoulli sampling (F_k). The weights and biases are updated by
          minimizing the differene F_0 - F_k.

4) Inference
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users

"""

# import libraries
import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt

import tensorflow as tf
import logging

import time as tm

# import default parameters
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    PREDICTION_COL,
)


# for logging
log = logging.getLogger(__name__)


class RBM:

    # initialize class parameters
    def __init__(
        self,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL,
        hidden_units=500,
        keep_prob=0.7,
        init_stdv=0.1,
        learning_rate=0.004,
        minibatch_size=100,
        training_epoch=20,
        display_epoch=10,
        cd_protocol=[50, 70, 80, 90, 100],
        save_path=None,
        debug=False,
        with_metrics=False,
    ):

        # pandas DF parameters
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        self.col_item = col_item
        self.col_user = col_user

        # RBM parameters
        self.Nh_ = hidden_units  # number of hidden units
        self.keep = keep_prob  # keep probability for dropout regularization
        self.std = (
            init_stdv
        )  # standard deviation used to initialize the weights matrices
        self.alpha = (
            learning_rate
        )  # learning rate used in the update method of the optimizer

        # size of the minibatch used in the random minibatches training; setting to 1 correspods to
        # stochastic gradient descent, and it is considerably slower.Good performance is achieved
        # for a size of ~100.
        self.minibatch = minibatch_size
        self.epochs = training_epoch + 1  # number of epochs used to train the model
        self.display = (
            display_epoch
        )  # number of epochs to show the mse error during training

        self.cd_protol = (
            cd_protocol
        )  # protocol to increase Gibbs sampling's step. Array containing the
        # percentage of the total training epoch when the step increases by 1

        # Options to save the model for future use
        self.save_path = save_path

        self.debug = (
            debug
        )  # if true, functions print their control paramters and/or outputs
        self.with_metrics = with_metrics  # compute msre and accuracy during training

        # Initialize the start time
        self.start_time = None

        log.info("TensorFlow version: {}".format(tf.__version__))

    # =========================
    # Helper functions
    # ========================

    # stateful time function
    def time(self):
        """
        Time a particular section of the code - call this once to set the state somewhere
        in the code, then call it again to return the elapsed time since last call.
        Call again to set the time and so on...

        Returns:
             if timer started time in seconds since the last time time function was called
        """

        if self.start_time is None:
            self.start_time = tm.time()
            return False

        else:
            answer = tm.time() - self.start_time
            # reset state
            self.start_time = None
            return answer

        # else:
        #    return None

    # Binomial sampling
    def B_sampling(self, pr):

        """
        Binomial sampling of hidden units activations using a rejection method.

        Args:
            pr (tensor, float32): input conditional probability
            g  (np.array, float32):  uniform probability used for comparison

        Returns:
            h_sampled (tensor, float32): sampled units. The value is 1 if pr>g and 0 otherwise.

        Basic mechanics:
            1) Extract a random number from a uniform distribution (g) and compare it with
                the unit's probability (pr)

            2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
               the relu function.

        """

        np.random.seed(1)

        # sample from a Bernoulli distribution with same dimensions as input distribution
        g = tf.convert_to_tensor(np.random.uniform(size=pr.shape[1]), dtype=tf.float32)

        # sample the value of the hidden units
        h_sampled = tf.nn.relu(tf.sign(pr - g))

        return h_sampled

    # Multinomial sampling
    def M_sampling(self, pr):

        """
        Multinomial Sampling of ratings

        Args:
            pr (tensor, float32): a distributions of shape (m, n, r), where m is the number of examples, n the number
                 of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.
            f (tensor, float32): normalized, uniform probability used for comparison.

        Returns:
            v_samp (tensor, float32): an (m,n) tensor of sampled rankings from 1 to r .

        Basic mechanics:
                For r classes, we sample r binomial distributions using the rejection method. This is possible
                since each class is statistically independent from the other. Note that this is the same method
                used in numpy's random.multinomial() function.

                1) extract a size r array of random numbers from a uniform distribution (g). As pr is normalized,
                   we need to normalize g as well.

                2) For each user and item, compare pr with the reference distribution. Note that the latter needs
                   to be the same for ALL the user/item pairs in the dataset, as by assumptions they are sampled
                   from a common distribution.

        """

        np.random.seed(1)

        g = np.random.uniform(size=pr.shape[2])  # sample from a uniform distribution
        f = tf.convert_to_tensor(
            g / g.sum(), dtype=tf.float32
        )  # normalize and convert to tensor

        samp = tf.nn.relu(tf.sign(pr - f))  # apply rejection method
        v_samp = tf.cast(
            tf.argmax(samp, axis=2) + 1, "float32"
        )  # select sampled element

        return v_samp

    # Multinomial distribution
    def Pm(self, phi):

        """
        Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi: linear combination of values of the previous layer
            r  : rating scale, corresponding to the number of classes

        Returns:
            pr: a tensor of shape (r, m, Nv) . This needs to be reshaped as (m, Nv, r) in the last step
                to allow for faster sampling when used in the Multinomial function.

        """

        num = [
            tf.exp(tf.multiply(tf.constant(k, dtype="float32"), phi))
            for k in range(1, self.r_ + 1)
        ]
        den = tf.reduce_sum(num, axis=0)

        pr = tf.div(num, den)

        return tf.transpose(pr, perm=[1, 2, 0])

    # Free energy
    def free_energy(self, x):

        """
        Free energy of the visible units given the hidden units. Since the sum is over the hidden units' states, the
        functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x: This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            F: Free energy of the model
        """

        bias = -tf.reduce_sum(tf.matmul(x, tf.transpose(self.bv)))

        phi_x = tf.matmul(x, self.w) + self.bh
        f = -tf.reduce_sum(tf.nn.softplus(phi_x))

        F = bias + f  # free energy density per training example

        return F

    # ==================================
    # Define graph topology
    # ==================================

    # Initialize graph
    # with self.graph.as_default():

    # Initialize the placeholders for the visible units
    def placeholder(self):
        self.vu = tf.placeholder(shape=[None, self.Nv_], dtype="float32")

    # initialize the parameters of the model.
    def init_parameters(self):

        """
        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and
        two bias vectors to initialize.

        Arguments:
            Nv (int): number of visible units (input layer)
            Nh (int): number of hidden units (latent variables of the model)

        Returns:
            w (tensor, float32): (Nv, Nh) correlation matrix initialized by sampling from a normal distribution with
               zero mean and given variance init_stdv.
           bv (tensor, float32): (1, Nv) visible units' bias, initialized to zero.
           bh (tensor, float32): (1, Nh) hidden units' bias, initiliazed to zero.

        """

        tf.set_random_seed(1)  # set the seed for the random number generator

        with tf.variable_scope("Network_parameters"):

            self.w = tf.get_variable(
                "weight",
                [self.Nv_, self.Nh_],
                initializer=tf.random_normal_initializer(stddev=self.std, seed=1),
                dtype="float32",
            )
            self.bv = tf.get_variable(
                "v_bias",
                [1, self.Nv_],
                initializer=tf.zeros_initializer(),
                dtype="float32",
            )
            self.bh = tf.get_variable(
                "h_bias",
                [1, self.Nh_],
                initializer=tf.zeros_initializer(),
                dtype="float32",
            )

    # ===================
    # Sampling
    # ===================

    """
    Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
    to initialize the two conditional probabilities:

    P(h|phi_v) --> returns the probability that the i-th hidden unit is active
    P(v|phi_h) --> returns the probability that the  i-th visible unit is active

    """

    # sample the hidden units given the visibles
    def sample_h(self, vv):

        """
        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv (tensor, float32): visible units

        Returns:
            phv (tensor, float32): activation probability of the hidden unit
            h_ (tensor, float32): sampled value of the hidden unit from a Bernoulli distributions having success probability phv

        """

        with tf.name_scope("sample_hidden_units"):

            phi_v = tf.matmul(vv, self.w) + self.bh  # create a linear combination
            phv = tf.nn.sigmoid(phi_v)  # conditional probability of h given v
            phv_reg = tf.nn.dropout(phv, self.keep)

            # Sampling
            h_ = self.B_sampling(
                phv_reg
            )  # obtain the value of the hidden units via Bernoulli sampling

        return phv, h_

    # sample the visible units given the hidden
    def sample_v(self, h):

        """
        Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN (negative phase)
        Each visible unit can take values in [1,r], while the zero is reserved for missing data; as such the value of the
        hidden unit is sampled from a multinomial distribution.

        Args:
            h (tensor, float32): visible units

        Returns:
            pvh (tensor, float32): activation probability of the visible unit given the hidden
            v_ (tensor, float32): sampled value of the visible unit from a Multinomial distributions having success probability pvh.

        Basic mechanics:
           1) For every training example we first sample Nv Multinomial distributions. The result is of the form [0,1,0,0,0,...,0]
              where the index of the 1 element corresponds to the rth rating. The index is extracted using the argmax function and
              we need to add 1 at the end since array indeces starts from 0.

           2) Selects only those units that have been sampled. During the training phase it is important to not use the reconstructed
              inputs, so we beed to enforce a zero value in the reconstructed ratings in the same position as the original input.

        """

        with tf.name_scope("sample_visible_units"):

            phi_h = tf.matmul(h, tf.transpose(self.w)) + self.bv  # linear combination
            pvh = self.Pm(phi_h)  # conditional probability of v given h

            # Sampling (modify here )
            v_tmp = self.M_sampling(pvh)  # sample the value of the visible units

            mask = tf.equal(self.v, 0)  # selects the inactive units in the input vector
            v_ = tf.where(
                mask, x=self.v, y=v_tmp
            )  # enforce inactive units in the reconstructed vector

        return pvh, v_

    # =======================
    # Training ops
    # =======================
    """
    Training in generative models takes place in two steps:

    1) Gibbs sampling
    2) Gradient evaluation and parameters update

    This estimate is later used in the weight update step by minimizing the distance between the
    model and the empirical free energy. Note that while the unit's configuration space is sampled,
    the weights are determined via maximum likelihood (saddle point).
    """

    # 1) Gibbs Sampling

    def G_sampling(self):

        """
        Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary RBM we need to
        impose that unseen movies stay as such, i.e. the sampling phase should not modify the elelments where v =0.

        Args:
            k (scalar, integer): iterator. Number of sampling steps
            v (tensor, float32): visible units

        Returns:
            h_k (tensor, float32): sampled value of the hidden unit at step  k
            v_k (tensor, float32): sampled value of the visible unit at step k

        """

        with tf.name_scope("gibbs_sampling"):

            self.v_k = (
                self.v
            )  # initialize the value of the visible units at step k=0 on the data

            if self.debug:
                print("CD step", self.k)

            for i in range(self.k):  # k_sampling
                _, h_k = self.sample_h(self.v_k)
                _, self.v_k = self.sample_v(h_k)

    # 2) Contrastive divergence
    def Losses(self, vv):

        """
        Loss functions

        Args:
            v (tensor, float32): empirical input
            v_k (tensor, float32): sampled visible units at step k

        Returns:
            obj (tensor, float32): objective function of Contrastive divergence, that is the difference between the
                 free energy clamped on the data (v) and the model Free energy (v_k)

        """

        with tf.variable_scope("losses"):

            obj = self.free_energy(vv) - self.free_energy(self.v_k)

        return obj

    def Gibbs_protocol(self, i):
        """
        Gibbs protocol

        Args:
            i (scalar, integer): current epoch in the loop

        Returns: G_sampling --> v_k (tensor, float32) evaluated at k steps

        Basic mechanics:
            If the current epoch i is in the interval specified in the training protocol cd_protocol_,
            the number of steps in Gibbs sampling (k) is incremented by one and G_sampling is updated
            accordingly.

        """

        with tf.name_scope("gibbs_protocol"):

            per = (i / self.epochs) * 100  # current percentage of the total #epochs

            if per != 0:
                if per >= self.cd_protol[self.l] and per <= self.cd_protol[self.l + 1]:
                    self.k += 1
                    self.l += 1
                    self.G_sampling()

            if self.debug:
                log.info("percentage of epochs covered so far %f2" % (per))

    # ================================================
    # model performance (online metrics)
    # ================================================

    # Inference
    def infere(self):

        """
        Prediction: A training example is used to activate the hidden unit that in turns produce new ratings for the
                        visible units, both for the rated and unrated examples.

        Args:
            xtr (np.array, float32): example from dataset. This can be either the test/train set

        Returns:
            vp (tensor, float32) : inferred values
            pvh (tensor, float32): probability of observing v given h

        """

        with tf.name_scope("inference"):
            # predict a new value
            _, h_p = self.sample_h(self.v)
            pvh, vp = self.sample_v(h_p)

        return pvh, vp

    # Metrics
    def accuracy(self, vp):

        """
        Train/Test Mean average precision

        Evaluates MAP over the train/test set in online mode. Note that this needs to be evaluated on the rated
        items only

        Args:
            vp (tensor, float32): inferred output (Network prediction)

        Returns:
            ac_score (tensor, float32)=  1/m Sum_{mu=1}^{m} Sum{i=1}^Nv 1/s(i) I(v-vp = 0)_{mu,i},
            where m = Nusers, Nv = number of items = number of visible units and s(i) is the number of non-zero
            elements per row.

        """
        with tf.name_scope("accuracy"):

            # 1) define and apply the mask
            mask = tf.not_equal(self.v, 0)
            n_values = tf.reduce_sum(tf.cast(mask, "float32"), axis=1)

            # 2) Take the difference between the input data and the inferred ones. This value is zero whenever the two
            #   values coincides
            vd = tf.where(
                mask, x=tf.abs(tf.subtract(self.v, vp)), y=tf.ones_like(self.v)
            )

            # correct values: find the location where v = vp
            corr = tf.cast(tf.equal(vd, 0), "float32")

            # 3) evaluate the accuracy
            ac_score = tf.reduce_mean(tf.div(tf.reduce_sum(corr, axis=1), n_values))

        return ac_score

    def msr_error(self, vp):

        """
        Mean square root error

        Note that this needs to be evaluated on the rated items only

        Args:
            vp (tensor, float32): inferred output (Network prediction)

        Returns:
            err (tensor, float32): msr error

        Returns:

        """

        with tf.name_scope("msr_error"):

            mask = tf.not_equal(self.v, 0)  # selects only the rated items
            n_values = tf.reduce_sum(
                tf.cast(mask, "float32"), axis=1
            )  # number of rated items

            # evaluate the square difference between the inferred and the input data on the rated items
            e = tf.where(
                mask, x=tf.squared_difference(self.v, vp), y=tf.zeros_like(self.v)
            )

            # evaluate the msre
            err = tf.sqrt(
                tf.reduce_mean(tf.div(tf.reduce_sum(e, axis=1), n_values)) / 2
            )

        return err

    # =========================
    # Training ops
    # =========================

    def fit(self, xtr, xtst):

        """
        Fit method

        Main component of the algo; once instantiated, it generates the computational graph and performs
        model training

        Args:
            xtr (np.array, integers): the user/affinity matrix for the train set
            xtst (np.array, integers): the user/affinity matrix for the test set

        Returns:
            elapsed (scalar, float32): elapsed time during training

            optional:
            msre (scalar, float32): the mean square root error per training epoch
            precision (scalar, float32): the mean average precision over the entire dataset

        """

        # keep the position of the items in the train set so that they can be optionally exluded from recommendation
        self.seen_mask = np.not_equal(xtr, 0)

        # start timing the methos
        self.time()

        self.r_ = xtr.max()  # defines the rating scale, e.g. 1 to 5
        m, self.Nv_ = xtr.shape  # dimension of the input: m= N_users, Nv= N_items

        num_minibatches = int(m / self.minibatch)  # number of minibatches

        tf.reset_default_graph()
        # ----------------------Initialize all parameters----------------

        log.info("Creating the computational graph")

        # create the visible units placeholder
        self.placeholder()

        self.batch_size_ = tf.placeholder(tf.int64)

        # Create data pipeline
        self.dataset = tf.data.Dataset.from_tensor_slices(self.vu)
        self.dataset = self.dataset.shuffle(
            buffer_size=50, reshuffle_each_iteration=True, seed=1
        )  # randomize the batch
        self.dataset = self.dataset.batch(batch_size=self.batch_size_).repeat()

        self.iter = self.dataset.make_initializable_iterator()
        self.v = self.iter.get_next()

        # initialize Network paramters
        self.init_parameters()

        # --------------Sampling protocol for Gibbs sampling-----------------------------------
        self.k = 1  # initialize the G_sampling step
        self.l = 0  # initialize epoch_sample index
        # -------------------------Main algo---------------------------

        self.G_sampling()  # returns the sampled value of the visible units

        obj = self.Losses(self.v)  # objective function
        rate = self.alpha / self.minibatch  # learning rate rescaled by the batch size

        # Instantiate the optimizer
        opt = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=rate).minimize(
            loss=obj
        )

        pvh, vp = (
            self.infere()
        )  # sample the value of the visible units given the hidden. Also returns  the related probabilities

        if self.with_metrics:  # if true (default) returns evaluation metrics
            Mse_train = []  # Lists to collect the metrics across epochs
            # Metrics
            Mserr = self.msr_error(self.v_k)
            Clacc = self.accuracy(self.v_k)

        if self.save_path != None:  # save the model to file
            saver = tf.train.Saver()

        init_g = (
            tf.global_variables_initializer()
        )  # Initialize all variables in the graph

        # Config GPU memory
        Config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        Config.gpu_options.per_process_gpu_memory_fraction = 0.5

        # Start TF training session on default graph
        self.sess = tf.Session(config=Config)
        self.sess.run(init_g)

        self.sess.run(
            self.iter.initializer,
            feed_dict={self.vu: xtr, self.batch_size_: self.minibatch},
        )

        if (
            self.with_metrics
        ):  # this condition is for benchmarking, remove for production

            # start loop over training epochs
            for i in range(self.epochs):

                epoch_tr_err = 0  # initialize the training error for each epoch to zero

                self.Gibbs_protocol(
                    i
                )  # updates the number of sampling steps in Gibbs sampling

                for l in range(num_minibatches):

                    _, batch_err = self.sess.run([opt, Mserr])

                    epoch_tr_err += (
                        batch_err / num_minibatches
                    )  # average msr error per minibatch

                if i % self.display == 0:
                    log.info("training epoch %i rmse Train %f" % (i, epoch_tr_err))

                # write metrics across epochs
                Mse_train.append(epoch_tr_err)  # mse training error per training epoch

            # Evaluates precision on the train and test set
            precision_train = self.sess.run(Clacc)

            self.sess.run(
                self.iter.initializer,
                feed_dict={self.vu: xtst, self.batch_size_: xtst.shape[0]},
            )
            precision_test = self.sess.run(Clacc)
            rmse_test = self.sess.run(Mserr)

            elapsed = self.time()

            log.info("done training, Training time %f2" % elapsed)

            # Display training error as a function of epochs
            plt.plot(Mse_train, label="train")
            plt.ylabel("msr_error", size="x-large")
            plt.xlabel("epochs", size="x-large")
            plt.legend(ncol=1)

            # Final precision scores
            log.info("Train set accuracy %f2" % precision_train)
            log.info("Test set accuracy %f2" % precision_test)

        else:

            # start loop over training epochs
            for i in range(self.epochs):

                self.Gibbs_protocol(
                    i
                )  # updates the number of sampling steps in Gibbs sampling

                for l in range(num_minibatches):

                    _ = self.sess.run(opt)

            elapsed = self.time()

            log.info("done training, Training time %f2" % elapsed)

        # --------------Save learning parameters and close session----------------------------
        if self.save_path != None:  # save the model to specified path
            saver.save(self.sess, self.save_path_ + "/rbm_model_saver.ckpt")

        return elapsed

    # =========================
    # Inference modules
    # =========================

    def eval_out(self):

        """
        Implement multinomial sampling from a trained model

        """

        # Sampling
        _, h_ = self.sample_h(self.vu)  # sample h

        # sample v
        phi_h = (
            tf.transpose(tf.matmul(self.w, tf.transpose(h_))) + self.bv
        )  # linear combination
        pvh_ = self.Pm(phi_h)  # conditional probability of v given h

        v_ = self.M_sampling(pvh_)  # sample the value of the visible units

        return v_, pvh_

    def recommend_k_items(self, x, top_k=10, remove_seen=True):

        """
        Returns the top-k items ordered by a relevancy score.

        Args:
            x (np.array, int32): input user/affinity matrix. Note that this can be a single vector, i.e. the ratings
            of a single user.
            top_k (scalar, int32): the number of items to recommend.

        Returns:
            top_scores (np.array, float32): a sparse matrix containing the top_k elements ordered by their score.

        Basic mechanics:
            The method can be called either within the same session or by restoring a previous session from file.
            If save_path is defined, a graph is generated and then populated with the pre-trained values of the
            parameters. Otherwise, the default session used during training is used.

            The method samples new ratings from the learned joint distribution, together with their probabilities.
            The input x must have the same number of columns as the one used for training the model (i.e. the same
            number of items) but it can have an arbitrary number of rows (users).

            A recommendation score is evaluated by taking the element-wise product between the ratings and the
            associated probabilities. For example, we could have the following situation:

                    rating     probability     score
            item1     5           0.5          2.5
            item2     4           0.8          3.2

            then item2 will be recommended.

        """
        self.time()

        if (
            self.save_path != None
        ):  # if true, restore the computational graph from a trained session

            m, self.Nv_ = x.shape  # dimension of the input: m= N_users, Nv= N_items
            self.r_ = x.max()  # defines the rating scale, e.g. 1 to 5
            tf.reset_default_graph()

            self.placeholder()
            self.init_parameters()

            saver = tf.train.Saver()
            self.sess = tf.Session()

            saved_files = saver.restore(
                self.sess, self.save_path_ + "/rbm_model_saver.ckpt"
            )

        else:
            m, _ = x.shape  # dimension of the input: m= N_users, Nv= N_items

        v_, pvh_ = (
            self.eval_out()
        )  # evaluate the ratings and the associated probabilities

        # evaluate v_ and pvh_ on the input data
        vp, pvh = self.sess.run([v_, pvh_], feed_dict={self.vu: x})

        pv = np.max(
            pvh, axis=2
        )  # returns only the probabilities for the predicted ratings in vp

        # evaluate the score
        score = np.multiply(vp, pv)
        # elapsed time
        elapsed = self.time()

        log.info("Done recommending items, time %f2" % elapsed)

        # ----------------------Return the results as a P dataframe------------------------------------
        log.info("Extracting top %i elements" % top_k)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            vp[self.seen_mask] = 0
            pv[self.seen_mask] = 0
            score[self.seen_mask] = 0

        top_items = np.argpartition(-score, range(top_k), axis=1)[
            :, :top_k
        ]  # get the top k items
        score_c = score.copy()  # get a copy of the score matrix
        score_c[
            np.arange(score_c.shape[0])[:, None], top_items
        ] = 0  # set to zero the top_k elements

        top_scores = score - score_c  # set to zeros all elements other then the top_k

        return top_scores, elapsed

    def predict(self, x, maps):

        """
        Returns the inferred ratings. This method is similar to recommend_k_items() with the following exceptions:

        - It returns a matrix
        - It returns all the inferred ratings

        Args:
            x (np.array, int32): input user/affinity matrix. Note that this can be a single vector, i.e. the ratings
                                 of a single user.

        Returns:
            vp (np.array, int32): a matrix with the inferred ratings.
            elapsed (scalar, float32): elapsed time for predediction.

        Basic mechanics:
            The method can be called either within the same session or by restoring a previous session from file.
            If save_path is specifie, a graph is generated and then populated with the pre-trained values of the
            parameters. Otherwise, the default session used during training is used.

            The method samples new ratings from the learned joint distribution, together with their probabilities.
            The input x must have the same number of columns as the one used for training the model, i.e. the same
            number of items, but it can have an arbitrary number of rows (users).

        """

        self.time()

        if (
            self.save_model_
        ):  # if true, restore the computational graph from a trained session

            m, self.Nv_ = x.shape  # dimension of the input: m= N_users, Nv= N_items

            self.r_ = x.max()  # defines the rating scale, e.g. 1 to 5

            tf.reset_default_graph()

            self.placeholder()
            self.init_parameters()

            saver = tf.train.Saver()

            self.sess = tf.Session()
            saved_files = saver.restore(
                self.sess, self.save_path_ + "/rbm_model_saver.ckpt"
            )

        else:
            m, _ = x.shape  # dimension of the input: m= N_users, Nv= N_items

        v_, _ = self.eval_out()  # evaluate the ratings and the associated probabilities

        vp = self.sess.run(v_, feed_dict={self.vu: x})

        elapsed = self.time()

        log.info("Done inference, time %f2" % elapsed)

        return vp, elapsed
