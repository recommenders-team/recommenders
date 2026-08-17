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
    """Restricted Boltzmann Machine for collaborative filtering."""

    def __init__(
        self,
        possible_ratings,
        visible_units,
        hidden_units=500,
        init_stdv=0.1,
        seed=42,
    ):
        """Implementation of the multinomial (softmax) Restricted Boltzmann Machine for
        collaborative filtering in numpy/pandas/pytorch.

        Based on the article by Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton
        https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

        Following the paper, each visible unit is a **one-hot softmax unit** over the
        rating scale: an item rated ``l`` is encoded as a vector of ``n_ratings`` binary
        units with a single 1 in position ``l``. Consequently the weights are a rank 3
        tensor ``w[i, l, j]`` and the visible bias a matrix ``bv[i, l]``, i.e. every
        rating value of every item has its own weights and bias. This is what allows the
        model to represent an arbitrary rating distribution per item; a single scalar
        unit with ``p(v=l) ~ exp(l * phi)`` can only represent monotone distributions and
        cannot express, e.g., "most users rate this movie a 3".

        Only the properties that define the model itself (its architecture and how its
        parameters are initialized) are set here. Everything that controls *how the model
        is trained* (number of epochs, minibatch size, learning rate, dropout, Gibbs
        sampling protocol, metrics) is passed to :meth:`fit` instead.

        Basic mechanics:

        1) The model parameters (a weight tensor and two biases) are created when the RBM
        class is instantiated. For an item based recommender:
        visible units: The number n_visible of visible units equals the number of items,
        each of them being a softmax unit over n_ratings classes
        hidden units : hyperparameter to fix during training

        2) Gibbs Sampling:

        2.1) for each training epoch, the visible units are first clamped on the data

        2.2) The activation probability of the hidden units, given a linear combination of
        the visibles, is evaluated P(h=1|phi_v). The latter is then used to sample the
        value of the hidden units.

        2.3) The probability P(v=l|phi_h) is evaluated, where l=1,..,r are the ratings (e.g.
        r=5 for the movielens dataset). This is a categorical distribution over the r
        classes, from which we sample the value of v.

        2.4) This step is repeated k times, where k increases as optimization converges. It is
        essential to fix to zero the original unrated items during the all learning process.

        3) Optimization:
        The free energy of the visible units given the hidden is evaluated at the beginning (F_0)
        and after k steps of Gibbs sampling (F_k). The weights and biases are updated by
        minimizing the difference F_0 - F_k.

        4) Inference:
        Once the joint probability distribution P(v,h) is learned, this is used to generate ratings
        for unrated items for all users. The inferred rating of an item is the *expected* rating
        under the learned distribution, sum_l l * P(v=l|h), as in the paper.

        Args:
            possible_ratings (list or numpy.ndarray): Sorted list of the unique rating values
                (e.g. ``[1, 2, 3, 4, 5]``). Its length is the number of softmax classes
                of each visible unit.
            visible_units (int): Number of visible units, i.e. the number of items.
            hidden_units (int): Number of hidden units (latent variables of the model).
            init_stdv (float): Standard deviation used to initialize the weight tensor.
            seed (int): Random seed for reproducible parameter initialization and training.
        """

        super().__init__()

        # ----------------------Model properties---------------------------------
        self.n_hidden = hidden_units  # number of hidden units
        self.n_visible = visible_units  # number of items

        # standard deviation used to initialize the weights matrices
        self.stdv = init_stdv

        # Seed
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ----------------------Initializers-------------------------------------

        # create a sorted list of all the unique ratings (of float type)
        self.possible_ratings = possible_ratings
        # python-side list of the ratings, used when building the categorical distribution
        self._ratings_list = [float(r) for r in possible_ratings]
        self.n_ratings = len(self._ratings_list)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # lookup table mapping the integer index of a class to its float rating value.
        # It replaces the tf.lookup.StaticHashTable of the original implementation and is
        # also used, in reverse, to one-hot encode the input ratings.
        self.register_buffer(
            "ratings_lookup_table",
            torch.tensor(self._ratings_list, dtype=torch.float32),
        )

        self.init_parameters()

        # Dropout is disabled at inference (eval mode), so this default only matters if
        # fit() is never called; fit() overwrites it with the requested keep probability.
        self.keep = 1.0

        # Training state, populated by fit().
        self.optimizer = None

        self.to(self.device)

    def init_parameters(self):
        """Initialize the parameters of the model.

        The model has one softmax visible unit per item, so the weights are a rank 3
        tensor and the visible bias is a matrix, one entry per (item, rating) pair.

        Returns:
            torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter:
            - `w` of size (n_visible, n_ratings, n_hidden): correlation tensor initialized by sampling from a normal distribution with zero mean and given variance init_stdv.
            - `bv` of size (n_visible, n_ratings): visible units' bias, initialized to zero.
            - `bh` of size (1, n_hidden): hidden units' bias, initiliazed to zero.
        """
        self.w = nn.Parameter(
            torch.empty(
                self.n_visible, self.n_ratings, self.n_hidden, dtype=torch.float32
            )
        )
        nn.init.normal_(self.w, mean=0.0, std=self.stdv)

        self.bv = nn.Parameter(
            torch.zeros(self.n_visible, self.n_ratings, dtype=torch.float32)
        )
        self.bh = nn.Parameter(torch.zeros(1, self.n_hidden, dtype=torch.float32))

        # flag recording whether bv already holds the empirical rating frequencies. It
        # lives in the state dict so that loading a checkpoint and calling fit() again to
        # resume training does not overwrite the learned bias.
        self.register_buffer("bv_initialized", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def to_one_hot(self, x):
        """One-hot encode a user/affinity matrix of ratings.

        A rating ``l`` of item ``i`` becomes a vector with a single 1 in the position of
        ``l`` within ``possible_ratings``. Unrated items (encoded as 0 in the affinity
        matrix) become an all-zero vector, so they contribute neither to the hidden
        activations nor to the gradients, as required by the paper.

        Args:
            x (torch.Tensor, float32): (m, n_visible) matrix of ratings, 0 meaning unrated.

        Returns:
            torch.Tensor, torch.Tensor:
            - `v`: (m, n_visible, n_ratings) float32 one-hot encoding of the ratings.
            - `mask`: (m, n_visible, 1) float32 indicator of the rated items.
        """

        mask = (x > 0).to(torch.float32).unsqueeze(-1)

        # index of the rating value inside possible_ratings. bucketize returns, for an
        # exact match, the index of the matching boundary; unrated entries map to 0 and
        # are zeroed out by the mask right after.
        idx = torch.bucketize(x, self.ratings_lookup_table)
        idx = idx.clamp_(max=self.n_ratings - 1)

        v = F.one_hot(idx, num_classes=self.n_ratings).to(torch.float32)

        return v * mask, mask

    def expected_rating(self, pvh):
        """Expected value of a rating under the categorical distribution `pvh`.

        This is the deterministic estimator used at inference time, ``sum_l l * P(v=l|h)``.

        Args:
            pvh (torch.Tensor, float32): (m, n_visible, n_ratings) normalized distribution.

        Returns:
            torch.Tensor: (m, n_visible) float32 tensor of expected ratings.
        """

        return torch.matmul(pvh, self.ratings_lookup_table)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def binomial_sampling(self, pr):
        """Binomial sampling of hidden units activations using a rejection method.

        Basic mechanics:

        1) Extract a random number from a uniform distribution (g) and compare it with
        the unit's probability (pr)

        2) Choose 0 if pr<g, 1 otherwise. It is convenient to implement this condtion using
        the relu function.

        Args:
            pr (torch.Tensor, float32): Input conditional probability.

        Returns:
            torch.Tensor: Float32 tensor of sampled units. The value is 1 if pr>g and 0 otherwise.
        """

        # sample from a Bernoulli distribution with same dimensions as input distribution
        g = torch.rand_like(pr)

        # sample the value of the hidden units
        h_sampled = (pr > g).to(torch.float32)

        return h_sampled

    def multinomial_sampling(self, pr):
        """Categorical sampling of the ratings.

        Each (user, item) pair is an independent categorical draw over the r rating
        classes, performed with ``torch.multinomial``. Note that this is a genuine
        categorical sample: every pair gets its own random number, unlike a rejection
        method that compares the whole distribution against a single shared threshold.

        Args:
            pr (torch.Tensor, float32): A distribution of shape (m, n, r), where m is the number of examples, n the number
                of features and r the number of classes. pr needs to be normalized, i.e. sum_k p(k) = 1 for all m, at fixed n.

        Returns:
            torch.Tensor: An (m, n, r) float32 one-hot tensor of sampled ratings.
        """

        m, n, r = pr.shape

        # torch.multinomial works on 2D inputs, so the user and item axes are flattened
        idx = torch.multinomial(pr.reshape(-1, r), num_samples=1).reshape(m, n)

        return F.one_hot(idx, num_classes=r).to(torch.float32)

    def multinomial_distribution(self, phi):
        """Probability that unit v has value l given phi: P(v=l|phi).

        With one-hot softmax units this is simply a softmax over the r classes of each
        visible unit, each class having its own linear combination phi[i, l].

        Args:
            phi (torch.Tensor): (m, n_visible, n_ratings) linear combination of the values
                of the previous layer.

        Returns:
            torch.Tensor: A (m, n_visible, n_ratings) tensor of normalized probabilities.
        """

        return F.softmax(phi, dim=-1)

    # ------------------------------------------------------------------
    # Conditional distributions
    # ------------------------------------------------------------------

    def _phi_v(self, v):
        """Linear combination of the visible units feeding the hidden layer."""

        # (m, n_visible * n_ratings) @ (n_visible * n_ratings, n_hidden)
        return torch.matmul(
            v.reshape(v.shape[0], -1), self.w.reshape(-1, self.n_hidden)
        )

    def _phi_h(self, h):
        """Linear combination of the hidden units feeding the visible layer."""

        # (m, n_hidden) @ (n_hidden, n_visible * n_ratings)
        phi = torch.matmul(h, self.w.reshape(-1, self.n_hidden).t())

        return phi.reshape(-1, self.n_visible, self.n_ratings) + self.bv

    def free_energy(self, v):
        """Free energy of the visible units given the hidden units. Since the sum is over the hidden units'
        states, the functional form of the visible units Free energy is the same as the one for the binary model.

        It is averaged over the examples of the minibatch so that the scale of the
        gradients, and therefore the meaning of the learning rate, does not depend on the
        minibatch size.

        Args:
            v (torch.Tensor): One-hot encoded visible units. This can be either the sampled
                value of the visible units (v_k) or the input data.

        Returns:
            torch.Tensor: Free energy of the model.
        """

        bias = -torch.sum(v * self.bv, dim=(1, 2))

        f = -torch.sum(F.softplus(self._phi_v(v) + self.bh), dim=1)

        # free energy density per training example
        return torch.mean(bias + f)

    def sample_hidden_units(self, v):
        """Sampling: In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need
        to initialize the two conditional probabilities:

        P(h|phi_v) --> returns the probability that the i-th hidden unit is active

        P(v|phi_h) --> returns the probability that the  i-th visible unit is active

        Sample hidden units given the visibles. This can be thought of as a Forward pass step in a FFN

        Args:
            v (torch.Tensor, float32): (m, n_visible, n_ratings) one-hot encoded visible units.

        Returns:
            torch.Tensor, torch.Tensor:
            - `phv`: The activation probability of the hidden unit.
            - `h_`: The sampled value of the hidden unit from a Bernoulli distributions having success probability `phv`.
        """

        phi_v = self._phi_v(v) + self.bh  # create a linear combination
        phv = torch.sigmoid(phi_v)  # conditional probability of h given v
        # dropout regularization; only active while the module is in training mode
        phv_reg = F.dropout(phv, p=1 - self.keep, training=self.training)

        # Sampling
        h_ = self.binomial_sampling(
            phv_reg
        )  # obtain the value of the hidden units via Bernoulli sampling

        return phv, h_

    def sample_visible_units(self, h, mask):
        """Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN
        (negative phase). Each visible unit is a softmax over the rating scale [1, r], while an all-zero
        vector is reserved for missing data; as such the value of the visible unit is sampled from a
        categorical distribution.

        Basic mechanics:

        1) For every training example we sample n_visible categorical distributions. The result is of the
        form [0,1,0,0,0] where the index of the 1 element corresponds to the sampled rating.

        2) Selects only those units that have been rated in the input. During the training phase it is
        important to not reconstruct the unrated items, so the reconstruction is zeroed in the same
        positions as the original input.

        Args:
            h (torch.Tensor, float32): hidden units.
            mask (torch.Tensor, float32): (m, n_visible, 1) indicator of the items rated in the
                original input, used to mask the unrated items in the reconstruction.

        Returns:
            torch.Tensor, torch.Tensor:
            - `pvh`: The activation probability of the visible unit given the hidden.
            - `v_`: The sampled one-hot value of the visible unit from a categorical distribution having success probability `pvh`.
        """

        phi_h = self._phi_h(h)  # linear combination
        pvh = self.multinomial_distribution(
            phi_h
        )  # conditional probability of v given h

        # Sampling
        v_ = self.multinomial_sampling(pvh)  # sample the value of the visible units

        # enforce inactive units in the reconstructed vector
        return pvh, v_ * mask

    def gibbs_sampling(self, v0, mask, k):
        """Gibbs sampling: Determines an estimate of the model configuration via sampling. In the binary
        RBM we need to impose that unseen movies stay as such, i.e. the sampling phase should not modify
        the elements where v=0.

        Args:
            v0 (torch.Tensor, float32): one-hot visible units clamped on the data (step k=0).
            mask (torch.Tensor, float32): indicator of the items rated in the input.
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
            _, v_k = self.sample_visible_units(h_k, mask)

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

    def rmse(self, x, pvh, mask):
        """Root mean squared error of the reconstruction, evaluated only on the observed
        (rated) entries and against the expected rating, not a sampled one.

        Args:
            x (torch.Tensor, float32): (m, n_visible) empirical ratings.
            pvh (torch.Tensor, float32): (m, n_visible, n_ratings) reconstructed distribution.
            mask (torch.Tensor, float32): (m, n_visible, 1) indicator of the rated items.

        Returns:
            float: masked RMSE between the input and its reconstruction.
        """

        mask2d = mask.squeeze(-1)
        pred = self.expected_rating(pvh)
        mse = torch.sum(((x - pred) ** 2) * mask2d) / torch.sum(mask2d)

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
            x = xtr[idx]
            v, mask = self.to_one_hot(x)

            # negative phase: sample the model configuration (no gradient flows through
            # the sampling steps, so it is computed under no_grad to save memory)
            with torch.no_grad():
                v_k = self.gibbs_sampling(v, mask, self.k)

            obj = self.losses(v, v_k)

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if self.with_metrics:
                with torch.no_grad():
                    # one-step mean-field reconstruction, using the activation
                    # probabilities rather than a sample, so that the reported error
                    # measures the model and not the sampling noise
                    phv, _ = self.sample_hidden_units(v)
                    pvh = self.multinomial_distribution(self._phi_h(phv))
                    batch_err = self.rmse(x, pvh, mask)
                # average msr error per minibatch
                epoch_tr_err += batch_err / num_minibatches

        return epoch_tr_err

    def fit(
        self,
        xtr,
        training_epoch=100,
        minibatch_size=100,
        learning_rate=0.001,
        keep_prob=0.7,
        l2=0.01,
        sampling_protocol=[50, 70, 80, 90, 100],
        display_epoch=10,
        with_metrics=False,
        debug=False,
    ):
        """Fit method

        Training in generative models takes place in two steps:

        1) Gibbs sampling
        2) Gradient evaluation and parameters update

        This estimate is later used in the weight update step by minimizing the distance between the
        model and the empirical free energy. Note that while the unit's configuration space is sampled,
        the weights are determined via maximum likelihood (saddle point).

        All the arguments below control the training process only; the model architecture is
        fixed at construction time.

        Args:
            xtr (numpy.ndarray, integers): the user/affinity matrix for the train set.
            training_epoch (int): number of epochs used to train the model. Contrastive
                divergence converges slowly, so the model should be trained until the
                training rmse (``with_metrics=True``) flattens out; on the MovieLens
                datasets this takes of the order of a hundred epochs.
            minibatch_size (int): size of the minibatch used in training. Setting it to 1
                corresponds to stochastic gradient descent and is considerably slower. Good
                performance is achieved for a size of ~100.
            learning_rate (float): learning rate used by the optimizer. The contrastive
                divergence objective is averaged over the minibatch, so this value does not
                need to be rescaled by the minibatch size.
            keep_prob (float): keep probability for dropout regularization of the hidden units.
            l2 (float): L2 weight decay applied by the optimizer.
            sampling_protocol (list): percentages of the total training epochs at which the
                number of Gibbs sampling steps is incremented by one.
            display_epoch (int): number of epochs after which the training rmse is logged.
            with_metrics (bool): if True, compute the rmse during training (stored in
                ``self.rmse_train``).
            debug (bool): if True, functions print their control parameters and/or outputs.
        """

        # A minibatch larger than the train set yields zero minibatches per epoch, which
        # would silently return an untrained model. Fail fast instead.
        n_users = xtr.shape[0]
        if minibatch_size > n_users:
            raise ValueError(
                f"minibatch_size ({minibatch_size}) cannot be larger than the number of "
                f"users in the train set ({n_users}), otherwise no minibatch is created "
                f"and the model is never updated."
            )

        # store the training hyperparameters on the instance for the training helpers
        self.epochs = training_epoch  # number of epochs used to train the model
        self.minibatch = minibatch_size  # size of the training minibatches
        self.learning_rate = learning_rate  # optimizer learning rate
        self.keep = keep_prob  # keep probability for dropout regularization
        self.l2 = l2  # weight decay
        self.sampling_protocol = sampling_protocol  # Gibbs sampling step protocol
        self.display_epoch = display_epoch  # epochs between rmse logs
        self.with_metrics = with_metrics  # whether to compute rmse during training
        self.debug = debug  # verbose control parameters/outputs

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2
        )

        # move the training data to the compute device once
        xtr_t = torch.as_tensor(xtr, dtype=torch.float32, device=self.device)

        # initialize the visible bias to the log frequency of each rating value. This is
        # the maximum likelihood solution of a model with no hidden units and gives the
        # optimizer a sensible starting point instead of a uniform rating distribution.
        # Skipped when resuming the training of an already fitted model.
        if not bool(self.bv_initialized):
            self._init_visible_bias(xtr_t)

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

    def _init_visible_bias(self, xtr):
        """Initialize the visible bias to the empirical log-frequency of the ratings.

        Args:
            xtr (torch.Tensor, float32): the user/affinity matrix for the train set.
        """

        with torch.no_grad():
            counts = torch.zeros_like(self.bv)

            # accumulate in chunks to keep the one-hot encoding of a large train set out
            # of memory in one go
            for start in range(0, xtr.shape[0], 1000):
                v, _ = self.to_one_hot(xtr[start : start + 1000])
                counts += v.sum(dim=0)

            # Laplace smoothing, so that a rating value never seen for an item does not
            # produce a -inf bias
            freq = (counts + 1.0) / (counts.sum(dim=1, keepdim=True) + self.n_ratings)
            self.bv.copy_(torch.log(freq))
            self.bv_initialized.fill_(True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def eval_out(self, x):
        """Conditional distribution of the ratings given the input, for a trained model.

        The hidden units are set to their activation probabilities (mean field) rather
        than to a sample, and no masking is applied to the visible layer, so that the
        distribution is defined for the unrated items too. Both choices make inference
        deterministic.

        Args:
            x (torch.Tensor, float32): (m, n_visible) input user/affinity matrix.

        Returns:
            torch.Tensor, torch.Tensor:
            - `v`: (m, n_visible) the expected ratings.
            - `pvh`: (m, n_visible, n_ratings) the associated probabilities.
        """

        self.eval()  # disable dropout for inference

        v_in, _ = self.to_one_hot(x)

        phv, _ = self.sample_hidden_units(v_in)  # activation probability of h

        pvh = self.multinomial_distribution(self._phi_h(phv))

        return self.expected_rating(pvh), pvh

    def _infer(self, x, batch_size=1000):
        """Run :meth:`eval_out` over the users in chunks.

        Args:
            x (numpy.ndarray): (m, n_visible) input user/affinity matrix.
            batch_size (int): number of users evaluated at a time.

        Returns:
            numpy.ndarray, numpy.ndarray: the expected ratings and their distributions.
        """

        x = np.atleast_2d(np.asarray(x))

        v_chunks, p_chunks = [], []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                chunk = torch.as_tensor(
                    x[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                v_, pvh_ = self.eval_out(chunk)
                v_chunks.append(v_.cpu().numpy())
                p_chunks.append(pvh_.cpu().numpy())

        return np.concatenate(v_chunks, axis=0), np.concatenate(p_chunks, axis=0)

    def recommend_k_items(self, x, top_k=10, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.

        Basic mechanics:

        The method infers the ratings from the learned joint distribution. The input x must
        have the same number of columns as the one used for training the model (i.e. the same
        number of items) but it can have an arbitrary number of rows (users).

        The recommendation score of an item is its expected rating under the learned
        distribution, ``sum_l l * P(v=l|h)``. Note that x must be the *observed* ratings of
        the users (e.g. the train set); passing the held-out ratings leaks the ground truth
        into the input and produces meaningless ranking metrics.

        Args:
            x (numpy.ndarray, int32): input user/affinity matrix. Note that this can be a single vector, i.e. the ratings
                of a single user.
            top_k (scalar, int32): the number of items to recommend.
            remove_seen (bool): if True, items already rated in ``x`` are not recommended.

        Returns:
            numpy.ndarray, float:
            - A sparse matrix containing the top_k elements ordered by their score.
        """

        score, _ = self._infer(x)

        log.info("Extracting top %i elements" % top_k)

        if remove_seen:
            # if true, it removes the items already rated in the input by setting their
            # score to zero. The mask is derived from the input itself so that the method
            # also works on users that were not part of the train set.
            score = np.where(np.atleast_2d(np.asarray(x)) > 0, 0, score)

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

        The method infers the ratings from the learned joint distribution. The returned value
        of an item is its expected rating ``sum_l l * P(v=l|h)``, which is the minimum mean
        squared error estimator and, unlike a sample from the same distribution, is
        deterministic. The input x must have the same number of columns as the one used for
        training the model, i.e. the same number of items, but it can have an arbitrary number
        of rows (users).

        Args:
            x (numpy.ndarray, int32): Input user/affinity matrix. Note that this can be a single vector, i.e.
                the ratings of a single user.

        Returns:
            numpy.ndarray, float:
            - A matrix with the inferred ratings.
        """

        v_, _ = self._infer(x)

        return v_

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
