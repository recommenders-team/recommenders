# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import time as tm


log = logging.getLogger(__name__)


class RBM:
    """Restricted  Boltzmann Machine"""

    def __init__(
        self,
        hidden_units=500,
        keep_prob=0.7,
        init_stdv=0.1,
        learning_rate=0.004,
        minibatch_size=100,
        training_epoch=20,
        display_epoch=10,
        sampling_protocol=[50, 70, 80, 90, 100],
        debug=False,
        with_metrics=False,
        seed=42,
    ):
        """Implementation of a multinomial Restricted Boltzmann Machine for collaborative filtering
        in numpy/pandas/tensorflow

        Based on the article by Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton
        https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

        In this implementation we use multinomial units instead of the one-hot-encoded used in
        the paper.This means that the weights are rank 2 (matrices) instead of rank 3 tensors.

        Basic mechanics:

        1) A computational graph is created when the RBM class is instantiated;
        For an item based recommender this consists of:
        visible units: The number Nv of visible units equals the number of items
        hidden units : hyperparameter to fix during training

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

        4) Inference:
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users
        """

        # RBM parameters
        self.Nhidden = hidden_units  # number of hidden units
        self.keep = keep_prob  # keep probability for dropout regularization

        # standard deviation used to initialize the weights matrices
        self.stdv = init_stdv

        # learning rate used in the update method of the optimizer
        self.learning_rate = learning_rate

        # size of the minibatch used in the random minibatches training; setting to 1 correspoNds to
        # stochastic gradient descent, and it is considerably slower.Good performance is achieved
        # for a size of ~100.
        self.minibatch = minibatch_size
        self.epochs = training_epoch + 1  # number of epochs used to train the model

        # number of epochs to show the mse error during training
        self.display = display_epoch

        # protocol to increase Gibbs sampling's step. Array containing the
        # percentage of the total training epoch when the step increases by 1
        self.sampling_protocol = sampling_protocol

        # if true, functions print their control paramters and/or outputs
        self.debug = debug

        # if true, compute msre and accuracy during training
        self.with_metrics = with_metrics

        # Initialize the start time
        self.start_time = None

        # Seed
        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    def time(self):
        """Time a particular section of the code - call this once to set the state somewhere
        in the code, then call it again to return the elapsed time since last call.
        Call again to set the time and so on...

        Returns:
            float: if timer started time in seconds since the last time time function was called
        """

        if self.start_time is None:
            self.start_time = tm.time()
            return False
        else:
            answer = tm.time() - self.start_time
            # reset state
            self.start_time = None
            return answer

    def binomial_sampling(self, pr):
        """Binomial sampling of hidden units activations using a rejection method.

        Basic mechanics:

        1) Extract a random number from a uniform distribution (g) and compare it with
        the unit's probability (pr)

        2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
        the relu function.

        Args:
            pr (tf.Tensor, float32): Input conditional probability.
            g  (numpy.ndarray, float32):  Uniform probability used for comparison.

        Returns:
            tf.Tensor: Float32 tensor of sampled units. The value is 1 if pr>g and 0 otherwise.
        """

        # sample from a Bernoulli distribution with same dimensions as input distribution
        g = tf.convert_to_tensor(np.random.uniform(size=pr.shape[1]), dtype=tf.float32)

        # sample the value of the hidden units
        h_sampled = tf.nn.relu(tf.sign(pr - g))

        return h_sampled

    def multinomial_sampling(self, pr):
        """Multinomial Sampling of ratings

        Basic mechanics:
        For r classes, we sample r binomial distributions using the rejection method. This is possible
        since each class is statistically independent from the other. Note that this is the same method
        used in numpy's random.multinomial() function.

        1) extract a size r array of random numbers from a uniform distribution (g). As pr is normalized,
        we need to normalize g as well.

        2) For each user and item, compare pr with the reference distribution. Note that the latter needs
        to be the same for ALL the user/item pairs in the dataset, as by assumptions they are sampled
        from a common distribution.

        Args:
            pr (tf.Tensor, float32): A distributions of shape (m, n, r), where m is the number of examples, n the number
                 of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.
            f (tf.Tensor, float32): Normalized, uniform probability used for comparison.

        Returns:
            tf.Tensor: An (m,n) float32 tensor of sampled rankings from 1 to r.
        """
        g = np.random.uniform(size=pr.shape[2])  # sample from a uniform distribution
        f = tf.convert_to_tensor(
            g / g.sum(), dtype=tf.float32
        )  # normalize and convert to tensor

        samp = tf.nn.relu(tf.sign(pr - f))  # apply rejection method
        v_samp = tf.cast(
            tf.argmax(samp, axis=2) + 1, "float32"
        )  # select sampled element

        return v_samp

    def multinomial_distribution(self, phi):
        """Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi (tf.Tensor): linear combination of values of the previous layer
            r (float): rating scale, corresponding to the number of classes

        Returns:
            tf.Tensor:
            - A tensor of shape (r, m, Nv): This needs to be reshaped as (m, Nv, r) in the last step to allow for faster sampling when used in the multinomial function.

        """

        numerator = [
            tf.exp(tf.multiply(tf.constant(k, dtype="float32"), phi))
            for k in range(1, self.ratings + 1)
        ]

        denominator = tf.reduce_sum(numerator, axis=0)

        prob = tf.div(numerator, denominator)

        return tf.transpose(prob, perm=[1, 2, 0])

    def free_energy(self, x):
        """Free energy of the visible units given the hidden units. Since the sum is over the hidden units'
        states, the functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x (tf.Tensor): This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            tf.Tensor: Free energy of the model.
        """

        bias = -tf.reduce_sum(tf.matmul(x, tf.transpose(self.bv)))

        phi_x = tf.matmul(x, self.w) + self.bh
        f = -tf.reduce_sum(tf.nn.softplus(phi_x))

        F = bias + f  # free energy density per training example

        return F

    def placeholder(self):
        """Initialize the placeholders for the visible units"""
        self.vu = tf.placeholder(shape=[None, self.Nvisible], dtype="float32")

    def init_parameters(self):
        """Initialize the parameters of the model.

        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and
        two bias vectors to initialize.

        Args:
            Nv (int): number of visible units (input layer)
            Nh (int): number of hidden units (latent variables of the model)

        Returns:
            tf.Tensor, tf.Tensor, tf.Tensor:
            - `w` of size (Nv, Nh): correlation matrix initialized by sampling from a normal distribution with zero mean and given variance init_stdv.
            - `bv` of size (1, Nvisible): visible units' bias, initialized to zero.
            - `bh` of size (1, Nhidden): hidden units' bias, initiliazed to zero.
        """
        with tf.variable_scope("Network_parameters"):

            self.w = tf.get_variable(
                "weight",
                [self.Nvisible, self.Nhidden],
                initializer=tf.random_normal_initializer(
                    stddev=self.stdv, seed=self.seed
                ),
                dtype="float32",
            )

            self.bv = tf.get_variable(
                "v_bias",
                [1, self.Nvisible],
                initializer=tf.zeros_initializer(),
                dtype="float32",
            )

            self.bh = tf.get_variable(
                "h_bias",
                [1, self.Nhidden],
                initializer=tf.zeros_initializer(),
                dtype="float32",
            )

    def sample_hidden_units(self, vv):
        """Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
        to initialize the two conditional probabilities:

        P(h|phi_v) --> returns the probability that the i-th hidden unit is active

        P(v|phi_h) --> returns the probability that the  i-th visible unit is active

        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv (tf.Tensor, float32): visible units

        Returns:
            tf.Tensor, tf.Tensor:
            - `phv`: The activation probability of the hidden unit.
            - `h_`: The sampled value of the hidden unit from a Bernoulli distributions having success probability `phv`.
        """

        with tf.name_scope("sample_hidden_units"):

            phi_v = tf.matmul(vv, self.w) + self.bh  # create a linear combination
            phv = tf.nn.sigmoid(phi_v)  # conditional probability of h given v
            phv_reg = tf.nn.dropout(phv, self.keep)

            # Sampling
            h_ = self.binomial_sampling(
                phv_reg
            )  # obtain the value of the hidden units via Bernoulli sampling

        return phv, h_

    def sample_visible_units(self, h):
        """Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN
        (negative phase). Each visible unit can take values in [1,rating], while the zero is reserved
        for missing data; as such the value of the hidden unit is sampled from a multinomial distribution.

        Basic mechanics:

        1) For every training example we first sample Nv Multinomial distributions. The result is of the
        form [0,1,0,0,0,...,0] where the index of the 1 element corresponds to the rth rating. The index
        is extracted using the argmax function and we need to add 1 at the end since array indeces starts
        from 0.

        2) Selects only those units that have been sampled. During the training phase it is important to not
        use the reconstructed inputs, so we beed to enforce a zero value in the reconstructed ratings in
        the same position as the original input.

        Args:
            h (tf.Tensor, float32): visible units.

        Returns:
            tf.Tensor, tf.Tensor:
            - `pvh`: The activation probability of the visible unit given the hidden.
            - `v_`: The sampled value of the visible unit from a Multinomial distributions having success probability `pvh`.
        """

        with tf.name_scope("sample_visible_units"):

            phi_h = tf.matmul(h, tf.transpose(self.w)) + self.bv  # linear combination
            pvh = self.multinomial_distribution(
                phi_h
            )  # conditional probability of v given h

            # Sampling (modify here )
            v_tmp = self.multinomial_sampling(
                pvh
            )  # sample the value of the visible units

            mask = tf.equal(self.v, 0)  # selects the inactive units in the input vector

            v_ = tf.where(
                mask, x=self.v, y=v_tmp
            )  # enforce inactive units in the reconstructed vector

        return pvh, v_

    def gibbs_sampling(self):
        """Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary
        RBM we need to impose that unseen movies stay as such, i.e. the sampling phase should not modify
        the elements where v=0.

        Args:
            k (scalar, integer): iterator. Number of sampling steps.
            v (tf.Tensor, float32): visible units.

        Returns:
            tf.Tensor, tf.Tensor:
            - `h_k`: The sampled value of the hidden unit at step k, float32.
            - `v_k`: The sampled value of the visible unit at step k, float32.
        """

        with tf.name_scope("gibbs_sampling"):

            self.v_k = (
                self.v
            )  # initialize the value of the visible units at step k=0 on the data

            if self.debug:
                print("CD step", self.k)

            for i in range(self.k):  # k_sampling
                _, h_k = self.sample_hidden_units(self.v_k)
                _, self.v_k = self.sample_visible_units(h_k)

    def losses(self, vv):
        """Loss functions.

        Args:
            v (tf.Tensor, float32): empirical input
            v_k (tf.Tensor, float32): sampled visible units at step k

        Returns:
            object:
            - Objective function of Contrastive divergence: the difference between the free energy clamped on the data (v) and the model Free energy (v_k).
        """

        with tf.variable_scope("losses"):
            obj = self.free_energy(vv) - self.free_energy(self.v_k)

        return obj

    def gibbs_protocol(self, i):
        """Gibbs protocol.

        Basic mechanics:

        If the current epoch i is in the interval specified in the training protocol,
        the number of steps in Gibbs sampling (k) is incremented by one and gibbs_sampling is updated
        accordingly.

        Args:
            i (int): Current epoch in the loop
        """

        with tf.name_scope("gibbs_protocol"):

            epoch_percentage = (
                i / self.epochs
            ) * 100  # current percentage of the total #epochs

            if epoch_percentage != 0:
                if (
                    epoch_percentage >= self.sampling_protocol[self.l]
                    and epoch_percentage <= self.sampling_protocol[self.l + 1]
                ):
                    self.k += 1
                    self.l += 1
                    self.gibbs_sampling()

            if self.debug:
                log.info("percentage of epochs covered so far %f2" % (epoch_percentage))

    def accuracy(self, vp):
        """Train/Test Mean average precision

        Evaluates MAP over the train/test set in online mode. Note that this needs to be evaluated on
        the rated items only.

        :math:`acc = 1/m \sum_{mu=1}^{m} \sum{i=1}^Nv 1/s(i) I(v-vp = 0)_{mu,i}`

        where `m = Nusers`, `Nv = number of items = number of visible units` and `s(i)` is the number of non-zero elements
        per row.

        Args:
            vp (tf.Tensor, float32): Inferred output (Network prediction)

        Returns:
            tf.Tensor: accuracy.

        """

        with tf.name_scope("accuracy"):

            # 1) define and apply the mask
            mask = tf.not_equal(self.v, 0)
            n_values = tf.reduce_sum(tf.cast(mask, "float32"), axis=1)

            # 2) Take the difference between the input data and the inferred ones. This value is zero whenever
            #    the two values coincides
            vd = tf.where(
                mask, x=tf.abs(tf.subtract(self.v, vp)), y=tf.ones_like(self.v)
            )

            # correct values: find the location where v = vp
            corr = tf.cast(tf.equal(vd, 0), "float32")

            # 3) evaluate the accuracy
            ac_score = tf.reduce_mean(tf.div(tf.reduce_sum(corr, axis=1), n_values))

        return ac_score

    def rmse(self, vp):
        """Root Mean Square Error

        Note that this needs to be evaluated on the rated items only

        Args:
            vp (tf.Tensor, float32): Inferred output (Network prediction)

        Returns:
            tf.Tensor: root mean square error.

        """

        with tf.name_scope("re"):

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

    def data_pipeline(self):
        """Define the data pipeline"""

        # placeholder for the batch_size
        self.batch_size = tf.placeholder(tf.int64)

        # Create the data pipeline for faster training
        self.dataset = tf.data.Dataset.from_tensor_slices(self.vu)

        self.dataset = self.dataset.shuffle(
            buffer_size=50, reshuffle_each_iteration=True, seed=self.seed
        )  # randomize the batch

        self.dataset = self.dataset.batch(batch_size=self.batch_size).repeat()

        # define iterator
        self.iter = self.dataset.make_initializable_iterator()
        self.v = self.iter.get_next()

    def init_metrics(self):
        """Initialize metrics"""

        if self.with_metrics:  # if true (default) returns evaluation metrics
            self.Rmse = self.rmse(self.v_k)
            self.Clacc = self.accuracy(self.v_k)

    def train_test_precision(self, xtst):
        """Evaluates precision on the train and test set

        Args:
            xtst (numpy.ndarray, integer32): The user/affinity matrix for the test set

        Returns:
            float, float: Precision on the train and test sets.
        """

        if self.with_metrics:

            precision_train = self.sess.run(self.Clacc)

            self.sess.run(
                self.iter.initializer,
                feed_dict={self.vu: xtst, self.batch_size: xtst.shape[0]},
            )

            precision_test = self.sess.run(self.Clacc)

        else:
            precision_train = None
            precision_test = None

        return precision_train, precision_test

    def display_metrics(self, Rmse_train, precision_train, precision_test):
        """Display training/test metrics and plots the rmse error as a function
        of the training epochs

        Args:
            Rmse_train (list, float32): Per epoch rmse on the train set.
            precision_train (float): Precision on the train set.
            precision_test  (float): Precision on the test set.
        """

        if self.with_metrics:

            # Display training error as a function of epochs
            plt.plot(Rmse_train, label="train")
            plt.ylabel("rmse", size="x-large")
            plt.xlabel("epochs", size="x-large")
            plt.legend(ncol=1)

            # Final precision scores
            log.info("Train set accuracy %f2" % precision_train)
            log.info("Test set accuracy %f2" % precision_test)

    def generate_graph(self):
        """Call the different RBM modules to generate the computational graph"""

        log.info("Creating the computational graph")

        self.placeholder()  # create the visible units placeholder
        self.data_pipeline()  # data_pipeline
        self.init_parameters()  # initialize Network parameters

        # --------------Initialize protocol for Gibbs sampling------------------
        log.info("Initialize Gibbs protocol")
        self.k = 1  # initialize the G_sampling step
        self.l = 0  # initialize epoch_sample index
        self.gibbs_sampling()  # returns the sampled value of the visible units

        # ---Instantiate loss function and optimizer----------------------------
        obj = self.losses(self.v)  # objective function

        rate = (
            self.learning_rate / self.minibatch
        )  # learning rate rescaled by the batch size

        self.opt = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=rate).minimize(
            loss=obj
        )  # Instantiate the optimizer

    def init_gpu(self):
        """Config GPU memory"""

        self.config_gpu = tf.ConfigProto(
            log_device_placement=True, allow_soft_placement=True
        )
        self.config_gpu.gpu_options.allow_growth = True  # dynamic memory allocation

    def init_training_session(self, xtr):
        """Initialize the TF session on training data

        Args:
            xtr (numpy.ndarray, int32): The user/affinity matrix for the train set.
        """

        init_graph = tf.global_variables_initializer()

        # Start TF training session on default graph
        self.sess = tf.Session(config=self.config_gpu)
        self.sess.run(init_graph)

        self.sess.run(
            self.iter.initializer,
            feed_dict={self.vu: xtr, self.batch_size: self.minibatch},
        )

    def batch_training(self, num_minibatches):
        """Perform training over input minibatches. If `self.with_metrics` is False,
        no online metrics are evaluated.

        Args:
            num_minibatches (scalar, int32): Number of training minibatches.

        Returns:
            float: Training error per single epoch. If `self.with_metrics` is False, this is zero.
        """

        epoch_tr_err = 0  # initialize the training error for each epoch to zero

        if self.with_metrics:

            for l in range(num_minibatches):  # minibatch loop
                _, batch_err = self.sess.run([self.opt, self.Rmse])
                # average msr error per minibatch
                epoch_tr_err += batch_err / num_minibatches

        else:
            for l in range(num_minibatches):  # minibatch loop
                _ = self.sess.run(self.opt)

        return epoch_tr_err

    def fit(self, xtr, xtst):
        """Fit method

        Training in generative models takes place in two steps:

        1) Gibbs sampling
        2) Gradient evaluation and parameters update

        This estimate is later used in the weight update step by minimizing the distance between the
        model and the empirical free energy. Note that while the unit's configuration space is sampled,
        the weights are determined via maximum likelihood (saddle point).

        Main component of the algo; once instantiated, it generates the computational graph and performs
        model training

        Args:
            xtr (numpy.ndarray, integers): the user/affinity matrix for the train set
            xtst (numpy.ndarray, integers): the user/affinity matrix for the test set

        Returns:
            float: elapsed time during training
        """

        # keep the position of the items in the train set so that they can be optionally exluded from recommendation
        self.seen_mask = np.not_equal(xtr, 0)
        self.time()
        self.ratings = xtr.max()  # obtain the rating scale, e.g. 1 to 5

        m, self.Nvisible = xtr.shape  # m= # users, Nvisible= # items
        num_minibatches = int(m / self.minibatch)  # number of minibatches

        tf.reset_default_graph()

        # ----------------------Initializers-------------------------------------
        self.generate_graph()
        self.init_metrics()
        self.init_gpu()
        self.init_training_session(xtr)

        Rmse_train = []  # List to collect the metrics across epochs

        # start loop over training epochs
        for i in range(self.epochs):

            self.gibbs_protocol(i)  # Gibbs sampling update
            epoch_tr_err = self.batch_training(num_minibatches)  # model train

            if self.with_metrics == True and i % self.display == 0:
                log.info("training epoch %i rmse %f" % (i, epoch_tr_err))

            Rmse_train.append(epoch_tr_err)  # mse training error per training epoch

        # optionally evaluate precision metrics
        precision_train, precision_test = self.train_test_precision(xtst)
        elapsed = self.time()

        log.info("done training, Training time %f2" % elapsed)

        self.display_metrics(Rmse_train, precision_train, precision_test)

        return elapsed

    def eval_out(self):
        """Implement multinomial sampling from a trained model"""

        # Sampling
        _, h = self.sample_hidden_units(self.vu)  # sample h

        # sample v
        phi_h = (
            tf.transpose(tf.matmul(self.w, tf.transpose(h))) + self.bv
        )  # linear combination
        pvh = self.multinomial_distribution(
            phi_h
        )  # conditional probability of v given h

        v = self.multinomial_sampling(pvh)  # sample the value of the visible units

        return v, pvh

    def recommend_k_items(self, x, top_k=10, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.

        Basic mechanics:

        The method samples new ratings from the learned joint distribution, together with their
        probabilities. The input x must have the same number of columns as the one used for training
        the model (i.e. the same number of items) but it can have an arbitrary number of rows (users).

        A recommendation score is evaluated by taking the element-wise product between the ratings and
        the associated probabilities. For example, we could have the following situation:

        .. code-block:: python

                    rating     probability     score
            item1     5           0.5          2.5
            item2     4           0.8          3.2

        then item2 will be recommended.

        Args:
            x (numpy.ndarray, int32): input user/affinity matrix. Note that this can be a single vector, i.e. the ratings
            of a single user.
            top_k (scalar, int32): the number of items to recommend.

        Returns:
            numpy.ndarray, float:
            - A sparse matrix containing the top_k elements ordered by their score.
            - The time taken to recommend k items.
        """

        self.time()
        # evaluate the ratings and the associated probabilities
        v_, pvh_ = self.eval_out()

        # evaluate v_ and pvh_ on the input data
        vp, pvh = self.sess.run([v_, pvh_], feed_dict={self.vu: x})
        # returns only the probabilities for the predicted ratings in vp
        pv = np.max(pvh, axis=2)

        # evaluate the score
        score = np.multiply(vp, pv)
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
        elapsed = self.time()

        log.info("Done recommending items, time %f2" % elapsed)
        return top_scores, elapsed

    def predict(self, x, maps):
        """Returns the inferred ratings. This method is similar to recommend_k_items() with the
        exceptions that it returns all the inferred ratings

        Basic mechanics:

        The method samples new ratings from the learned joint distribution, together with
        their probabilities. The input x must have the same number of columns as the one used
        for training the model, i.e. the same number of items, but it can have an arbitrary number
        of rows (users).

        Args:
            x (numpy.ndarray, int32): Input user/affinity matrix. Note that this can be a single vector, i.e.
            the ratings of a single user.

        Returns:
            numpy.ndarray, float:
            - A matrix with the inferred ratings.
            - The elapsed time for predediction.
        """

        self.time()

        v_, _ = self.eval_out()  # evaluate the ratings and the associated probabilities
        vp = self.sess.run(v_, feed_dict={self.vu: x})
        elapsed = self.time()

        log.info("Done inference, time %f2" % elapsed)

        return vp, elapsed
