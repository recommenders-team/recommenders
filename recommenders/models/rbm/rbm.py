# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class RBM(nn.Module):
    """Restricted Boltzmann Machine"""

    def __init__(
        self,
        possible_ratings,
        visible_units,
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
        in numpy/pandas/pytorch

        Based on the article by Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton
        https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

        In this implementation we use multinomial units instead of the one-hot-encoded used in
        the paper. This means that the weights are rank 2 (matrices) instead of rank 3 tensors.

        Basic mechanics:

        1) The model parameters (a weight matrix and two bias vectors) are created when the RBM
        class is instantiated. For an item based recommender:
        visible units: The number n_visible of visible units equals the number of items
        hidden units : hyperparameter to fix during training

        2) Gibbs Sampling:

        2.1) for each training epoch, the visible units are first clamped on the data

        2.2) The activation probability of the hidden units, given a linear combination of
        the visibles, is evaluated P(h=1|phi_v). The latter is then used to sample the
        value of the hidden units.

        2.3) The probability P(v=l|phi_h) is evaluated, where l=1,..,r are the ratings (e.g.
        r=5 for the movielens dataset). In general, this is a multinomial distribution,
        from which we sample the value of v.

        2.4) This step is repeated k times, where k increases as optimization converges. It is
        essential to fix to zero the original unrated items during the all learning process.

        3) Optimization:
        The free energy of the visible units given the hidden is evaluated at the beginning (F_0)
        and after k steps of Bernoulli sampling (F_k). The weights and biases are updated by
        minimizing the difference F_0 - F_k.

        4) Inference:
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users
        """

        super().__init__()

        # RBM parameters
        self.n_hidden = hidden_units  # number of hidden units
        self.keep = keep_prob  # keep probability for dropout regularization

        # standard deviation used to initialize the weights matrices
        self.stdv = init_stdv

        # learning rate used in the update method of the optimizer
        self.learning_rate = learning_rate

        # size of the minibatch used in the random minibatches training; setting to 1 corresponds to
        # stochastic gradient descent, and it is considerably slower. Good performance is achieved
        # for a size of ~100.
        self.minibatch = minibatch_size
        self.epochs = training_epoch + 1  # number of epochs used to train the model

        # number of epochs to show the mse error during training
        self.display_epoch = display_epoch

        # protocol to increase Gibbs sampling's step. Array containing the
        # percentage of the total training epoch when the step increases by 1
        self.sampling_protocol = sampling_protocol

        # if true, functions print their control parameters and/or outputs
        self.debug = debug

        # if true, compute msre and accuracy during training
        self.with_metrics = with_metrics

        # Seed
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.n_visible = visible_units  # number of items

        # ----------------------Initializers-------------------------------------

        # create a sorted list of all the unique ratings (of float type)
        self.possible_ratings = possible_ratings
        # python-side list of the ratings, used when building the multinomial distribution
        self._ratings_list = [float(r) for r in possible_ratings]
        self.n_ratings = len(self._ratings_list)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # lookup table mapping the integer index of a class to its float rating value.
        # It replaces the tf.lookup.StaticHashTable of the original implementation.
        self.register_buffer(
            "ratings_lookup_table",
            torch.tensor(self._ratings_list, dtype=torch.float32),
        )

        self.init_parameters()

        # optimizer is instantiated in fit(); mask of seen items is populated there too
        self.optimizer = None
        self.seen_mask = None

        self.to(self.device)

    def init_parameters(self):
        """Initialize the parameters of the model.

        This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and
        two bias vectors to initialize.

        Args:
            n_visible (int): number of visible units (input layer)
            n_hidden (int): number of hidden units (latent variables of the model)

        Returns:
            torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter:
            - `w` of size (n_visible, n_hidden): correlation matrix initialized by sampling from a normal distribution with zero mean and given variance init_stdv.
            - `bv` of size (1, n_visible): visible units' bias, initialized to zero.
            - `bh` of size (1, n_hidden): hidden units' bias, initiliazed to zero.
        """
        self.w = nn.Parameter(
            torch.empty(self.n_visible, self.n_hidden, dtype=torch.float32)
        )
        nn.init.normal_(self.w, mean=0.0, std=self.stdv)

        self.bv = nn.Parameter(torch.zeros(1, self.n_visible, dtype=torch.float32))
        self.bh = nn.Parameter(torch.zeros(1, self.n_hidden, dtype=torch.float32))

    def binomial_sampling(self, pr):
        """Binomial sampling of hidden units activations using a rejection method.

        Basic mechanics:

        1) Extract a random number from a uniform distribution (g) and compare it with
        the unit's probability (pr)

        2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
        the relu function.

        Args:
            pr (torch.Tensor, float32): Input conditional probability.
            g  (torch.Tensor, float32):  Uniform probability used for comparison.

        Returns:
            torch.Tensor: Float32 tensor of sampled units. The value is 1 if pr>g and 0 otherwise.
        """

        # sample from a Bernoulli distribution with same dimensions as input distribution
        g = torch.rand(pr.shape[1], dtype=torch.float32, device=pr.device)

        # sample the value of the hidden units
        h_sampled = F.relu(torch.sign(pr - g))

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
            pr (torch.Tensor, float32): A distributions of shape (m, n, r), where m is the number of examples, n the number
                 of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.
            f (torch.Tensor, float32): Normalized, uniform probability used for comparison.

        Returns:
            torch.Tensor: An (m,n) float32 tensor of sampled rankings from 1 to r.
        """
        g = torch.rand(pr.shape[2], dtype=torch.float32, device=pr.device)
        f = g / g.sum()  # normalize

        samp = F.relu(torch.sign(pr - f))  # apply rejection method

        # get integer index of the rating to be sampled
        v_argmax = torch.argmax(samp, dim=2)

        # lookup the rating using integer index
        v_samp = self.ratings_lookup_table[v_argmax].to(torch.float32)

        return v_samp

    def multinomial_distribution(self, phi):
        """Probability that unit v has value l given phi: P(v=l|phi)

        Args:
            phi (torch.Tensor): linear combination of values of the previous layer
            r (float): rating scale, corresponding to the number of classes

        Returns:
            torch.Tensor:
            - A tensor of shape (r, m, Nv): This needs to be reshaped as (m, Nv, r) in the last step to allow for faster sampling when used in the multinomial function.

        """

        numerator = torch.stack([torch.exp(k * phi) for k in self._ratings_list], dim=0)

        denominator = torch.sum(numerator, dim=0)

        prob = numerator / denominator

        return prob.permute(1, 2, 0)

    def free_energy(self, x):
        """Free energy of the visible units given the hidden units. Since the sum is over the hidden units'
        states, the functional form of the visible units Free energy is the same as the one for the binary model.

        Args:
            x (torch.Tensor): This can be either the sampled value of the visible units (v_k) or the input data

        Returns:
            torch.Tensor: Free energy of the model.
        """

        bias = -torch.sum(torch.matmul(x, self.bv.t()))

        phi_x = torch.matmul(x, self.w) + self.bh
        f = -torch.sum(F.softplus(phi_x))

        free_energy = bias + f  # free energy density per training example

        return free_energy

    def sample_hidden_units(self, vv):
        """Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
        to initialize the two conditional probabilities:

        P(h|phi_v) --> returns the probability that the i-th hidden unit is active

        P(v|phi_h) --> returns the probability that the  i-th visible unit is active

        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            vv (torch.Tensor, float32): visible units

        Returns:
            torch.Tensor, torch.Tensor:
            - `phv`: The activation probability of the hidden unit.
            - `h_`: The sampled value of the hidden unit from a Bernoulli distributions having success probability `phv`.
        """

        phi_v = torch.matmul(vv, self.w) + self.bh  # create a linear combination
        phv = torch.sigmoid(phi_v)  # conditional probability of h given v
        # dropout regularization; only active while the module is in training mode
        phv_reg = F.dropout(phv, p=1 - self.keep, training=self.training)

        # Sampling
        h_ = self.binomial_sampling(
            phv_reg
        )  # obtain the value of the hidden units via Bernoulli sampling

        return phv, h_

    def sample_visible_units(self, h, v0):
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
            h (torch.Tensor, float32): hidden units.
            v0 (torch.Tensor, float32): original (clamped) visible units, used to mask the
                unrated items in the reconstruction.

        Returns:
            torch.Tensor, torch.Tensor:
            - `pvh`: The activation probability of the visible unit given the hidden.
            - `v_`: The sampled value of the visible unit from a Multinomial distributions having success probability `pvh`.
        """

        phi_h = torch.matmul(h, self.w.t()) + self.bv  # linear combination
        pvh = self.multinomial_distribution(
            phi_h
        )  # conditional probability of v given h

        # Sampling
        v_tmp = self.multinomial_sampling(pvh)  # sample the value of the visible units

        mask = torch.eq(v0, 0)  # selects the inactive units in the input vector

        v_ = torch.where(
            mask, v0, v_tmp
        )  # enforce inactive units in the reconstructed vector

        return pvh, v_

    def gibbs_sampling(self, v0, k):
        """Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary
        RBM we need to impose that unseen movies stay as such, i.e. the sampling phase should not modify
        the elements where v=0.

        Args:
            v0 (torch.Tensor, float32): visible units clamped on the data (step k=0).
            k (int): number of sampling steps.

        Returns:
            torch.Tensor:
            - `v_k`: The sampled value of the visible unit at step k, float32.
        """

        v_k = v0  # initialize the value of the visible units at step k=0 on the data

        if self.debug:
            print("CD step", k)

        for _ in range(k):  # k_sampling
            _, h_k = self.sample_hidden_units(v_k)
            _, v_k = self.sample_visible_units(h_k, v0)

        return v_k

    def losses(self, vv, v_k):
        """Calculate contrastive divergence, which is the difference between
        the free energy clamped on the data (v) and the model Free energy (v_k).

        Args:
            vv (torch.Tensor, float32): empirical input
            v_k (torch.Tensor, float32): sampled visible units after k Gibbs steps

        Returns:
            torch.Tensor: contrastive divergence
        """

        obj = self.free_energy(vv) - self.free_energy(v_k)

        return obj

    def gibbs_protocol(self, i):
        """Gibbs protocol.

        Basic mechanics:

        If the current epoch i is in the interval specified in the training protocol,
        the number of steps in Gibbs sampling (k) is incremented by one.

        Args:
            i (int): Current epoch in the loop
        """

        epoch_percentage = (
            i / self.epochs
        ) * 100  # current percentage of the total #epochs

        if epoch_percentage != 0:
            if (
                self.l + 1 < len(self.sampling_protocol)
                and epoch_percentage >= self.sampling_protocol[self.l]
                and epoch_percentage <= self.sampling_protocol[self.l + 1]
            ):
                self.k += 1
                self.l += 1  # noqa: E741 ambiguous variable name 'l'

        if self.debug:
            log.info("percentage of epochs covered so far %f2" % (epoch_percentage))

    def rmse(self, v, v_k):
        """Root mean squared error evaluated only on the observed (non-zero) entries.

        Args:
            v (torch.Tensor, float32): empirical input
            v_k (torch.Tensor, float32): reconstructed visible units

        Returns:
            float: masked RMSE between the input and its reconstruction.
        """

        mask = (v > 0).to(torch.float32)
        mse = torch.sum(((v - v_k) ** 2) * mask) / torch.sum(mask)

        return torch.sqrt(mse).item()

    def batch_training(self, xtr):
        """Perform a single training epoch over the input minibatches. If `self.with_metrics`
        is False, no online metrics are evaluated.

        Args:
            xtr (torch.Tensor, float32): the user/affinity matrix for the train set.

        Returns:
            float: Training error for the epoch. If `self.with_metrics` is False, this is zero.
        """

        n_users = xtr.shape[0]
        num_minibatches = int(n_users / self.minibatch)

        # randomize the order of the training examples for this epoch
        perm = torch.randperm(n_users, device=xtr.device)

        epoch_tr_err = 0  # initialize the training error for each epoch to zero

        # minibatch loop
        for b in range(num_minibatches):
            idx = perm[b * self.minibatch : (b + 1) * self.minibatch]
            v = xtr[idx]

            # negative phase: sample the model configuration (no gradient flows through
            # the sampling steps, so it is computed under no_grad to save memory)
            with torch.no_grad():
                v_k = self.gibbs_sampling(v, self.k)

            obj = self.losses(v, v_k)

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if self.with_metrics:
                with torch.no_grad():
                    batch_err = self.rmse(v, v_k)
                # average msr error per minibatch
                epoch_tr_err += batch_err / num_minibatches

        return epoch_tr_err

    def fit(self, xtr):
        """Fit method

        Training in generative models takes place in two steps:

        1) Gibbs sampling
        2) Gradient evaluation and parameters update

        This estimate is later used in the weight update step by minimizing the distance between the
        model and the empirical free energy. Note that while the unit's configuration space is sampled,
        the weights are determined via maximum likelihood (saddle point).

        Args:
            xtr (numpy.ndarray, integers): the user/affinity matrix for the train set
        """

        # keep the position of the items in the train set so that they can be optionally exluded from recommendation
        self.seen_mask = np.not_equal(xtr, 0)

        # learning rate rescaled by the batch size
        rate = self.learning_rate / self.minibatch
        self.optimizer = torch.optim.Adam(self.parameters(), lr=rate)

        # move the training data to the compute device once
        xtr_t = torch.as_tensor(xtr, dtype=torch.float32, device=self.device)

        self.train()  # enable dropout during training

        # --------------Initialize protocol for Gibbs sampling------------------
        self.k = 1  # initialize the G_sampling step
        self.l = 0  # noqa: E741 initialize epoch_sample index

        rmse_train = []  # List to collect the metrics across epochs

        # start loop over training epochs
        for i in range(self.epochs):

            self.gibbs_protocol(i)  # Gibbs sampling update
            epoch_tr_err = self.batch_training(xtr_t)  # model train

            if self.with_metrics and i % self.display_epoch == 0:
                log.info("training epoch %i rmse %f" % (i, epoch_tr_err))

            rmse_train.append(epoch_tr_err)  # mse training error per training epoch

        self.rmse_train = rmse_train

    def eval_out(self, x):
        """Implement multinomial sampling from a trained model

        Args:
            x (torch.Tensor, float32): input user/affinity matrix.

        Returns:
            torch.Tensor, torch.Tensor:
            - `v`: the sampled ratings.
            - `pvh`: the associated probabilities.
        """

        self.eval()  # disable dropout for inference

        # Sampling
        _, h = self.sample_hidden_units(x)  # sample h

        # sample v
        phi_h = torch.matmul(h, self.w.t()) + self.bv  # linear combination
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
        """

        # evaluate the ratings and the associated probabilities on the input data
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_, pvh_ = self.eval_out(x_t)

        vp = v_.cpu().numpy()
        pvh = pvh_.cpu().numpy()
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

        score_c[np.arange(score_c.shape[0])[:, None], top_items] = (
            0  # set to zero the top_k elements
        )

        top_scores = score - score_c  # set to zeros all elements other then the top_k

        return top_scores

    def predict(self, x):
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
        """

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_, _ = self.eval_out(x_t)  # evaluate the ratings

        return v_.cpu().numpy()

    def save(self, file_path="./rbm_model.pt"):
        """Save model parameters to `file_path`

        This function saves the current model state dictionary to a specified path.

        Args:
            file_path (str): output file path for the RBM model checkpoint
                we will create a new directory if not existing.
        """

        f_path = Path(file_path)
        dir_name = f_path.parent

        # create the directory if it does not exist
        os.makedirs(dir_name, exist_ok=True)

        # save trained model
        torch.save(self.state_dict(), file_path)

    def load(self, file_path="./rbm_model.pt"):
        """Load model parameters for further use.

        This function loads a saved model state dictionary.

        Args:
            file_path (str): file path for RBM model checkpoint
        """

        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
