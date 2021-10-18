# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommenders.evaluation.python_evaluation import ndcg_at_k

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback


class LossHistory(Callback):
    """This class is used for saving the validation loss and the training loss per epoch."""

    def on_train_begin(self, logs={}):
        """Initialise the lists where the loss of training and validation will be saved."""
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        """Save the loss of training and validation set at the end of each epoch."""
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


class Metrics(Callback):
    """Callback function used to calculate the NDCG@k metric of validation set at the end of each epoch.
    Weights of the model with the highest NDCG@k value is saved."""

    def __init__(self, model, val_tr, val_te, mapper, k, save_path=None):

        """Initialize the class parameters.

        Args:
            model: trained model for validation.
            val_tr (numpy.ndarray, float): the click matrix for the validation set training part.
            val_te (numpy.ndarray, float): the click matrix for the validation set testing part.
            mapper (AffinityMatrix): the mapper for converting click matrix to dataframe.
            k (int): number of top k items per user (optional).
            save_path (str): Default path to save weights.
        """
        # Model
        self.model = model

        # Initial value of NDCG
        self.best_ndcg = 0.0

        # Validation data: training and testing parts
        self.val_tr = val_tr
        self.val_te = val_te

        # Mapper for converting from sparse matrix to dataframe
        self.mapper = mapper

        # Top k items to recommend
        self.k = k

        # Options to save the weights of the model for future use
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        """Initialise the list for validation NDCG@k."""
        self._data = []

    def recommend_k_items(self, x, k, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.
        Obtained probabilities are used as recommendation score.

        Args:
            x (numpy.ndarray, int32): input click matrix.
            k (scalar, int32): the number of items to recommend.

        Returns:
            numpy.ndarray: A sparse matrix containing the top_k elements ordered by their score.

        """
        # obtain scores
        score = self.model.predict(x)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0

        # get the top k items
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]

        # get a copy of the score matrix
        score_c = score.copy()

        # set to zero the k elements
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0

        # set to zeros all elements other then the k
        top_scores = score - score_c

        return top_scores

    def on_epoch_end(self, batch, logs={}):
        """At the end of each epoch calculate NDCG@k of the validation set.

        If the model performance is improved, the model weights are saved.
        Update the list of validation NDCG@k by adding obtained value

        """
        # recommend top k items based on training part of validation set
        top_k = self.recommend_k_items(x=self.val_tr, k=self.k, remove_seen=True)

        # convert recommendations from sparse matrix to dataframe
        top_k_df = self.mapper.map_back_sparse(top_k, kind="prediction")
        test_df = self.mapper.map_back_sparse(self.val_te, kind="ratings")

        # calculate NDCG@k
        NDCG = ndcg_at_k(test_df, top_k_df, col_prediction="prediction", k=self.k)

        # check if there is an improvement in NDCG, if so, update the weights of the saved model
        if NDCG > self.best_ndcg:
            self.best_ndcg = NDCG

            # save the weights of the optimal model
            if self.save_path is not None:
                self.model.save(self.save_path)

        self._data.append(NDCG)

    def get_data(self):
        """Returns a list of the NDCG@k of the validation set metrics calculated
        at the end of each epoch."""
        return self._data


class AnnealingCallback(Callback):
    """This class is used for updating the value of β during the annealing process.
    When β reaches the value of anneal_cap, it stops increasing."""

    def __init__(self, beta, anneal_cap, total_anneal_steps):

        """Constructor

        Args:
            beta (float): current value of beta.
            anneal_cap (float): maximum value that beta can reach.
            total_anneal_steps (int): total number of annealing steps.
        """
        # maximum value that beta can take
        self.anneal_cap = anneal_cap

        # initial value of beta
        self.beta = beta

        # update_count used for calculating the updated value of beta
        self.update_count = 0

        # total annealing steps
        self.total_anneal_steps = total_anneal_steps

    def on_train_begin(self, logs={}):
        """Initialise a list in which the beta value will be saved at the end of each epoch."""
        self._beta = []

    def on_batch_end(self, epoch, logs={}):
        """At the end of each batch the beta should is updated until it reaches the values of anneal cap."""
        self.update_count = self.update_count + 1

        new_beta = min(
            1.0 * self.update_count / self.total_anneal_steps, self.anneal_cap
        )

        K.set_value(self.beta, new_beta)

    def on_epoch_end(self, epoch, logs={}):
        """At the end of each epoch save the value of beta in _beta list."""
        tmp = K.eval(self.beta)
        self._beta.append(tmp)

    def get_data(self):
        """Returns a list of the beta values per epoch."""
        return self._beta


class Mult_VAE:
    """Multinomial Variational Autoencoders (Multi-VAE) for Collaborative Filtering implementation

    :Citation:

        Liang, Dawen, et al. "Variational autoencoders for collaborative filtering."
        Proceedings of the 2018 World Wide Web Conference. 2018.
        https://arxiv.org/pdf/1802.05814.pdf
    """

    def __init__(
        self,
        n_users,
        original_dim,
        intermediate_dim=200,
        latent_dim=70,
        n_epochs=400,
        batch_size=100,
        k=100,
        verbose=1,
        drop_encoder=0.5,
        drop_decoder=0.5,
        beta=1.0,
        annealing=False,
        anneal_cap=1.0,
        seed=None,
        save_path=None,
    ):

        """Constructor

        Args:
            n_users (int): Number of unique users in the train set.
            original_dim (int): Number of unique items in the train set.
            intermediate_dim (int): Dimension of intermediate space.
            latent_dim (int): Dimension of latent space.
            n_epochs (int): Number of epochs for training.
            batch_size (int): Batch size.
            k (int): number of top k items per user.
            verbose (int): Whether to show the training output or not.
            drop_encoder (float): Dropout percentage of the encoder.
            drop_decoder (float): Dropout percentage of the decoder.
            beta (float): a constant parameter β in the ELBO function,
                  when you are not using annealing (annealing=False)
            annealing (bool): option of using annealing method for training the model (True)
                  or not using annealing, keeping a constant beta (False)
            anneal_cap (float): maximum value that beta can take during annealing process.
            seed (int): Seed.
            save_path (str): Default path to save weights.
        """
        # Seed
        self.seed = seed
        np.random.seed(self.seed)

        # Parameters
        self.n_users = n_users
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose

        # Compute samples per epoch
        self.number_of_batches = self.n_users // self.batch_size

        # Annealing parameters
        self.anneal_cap = anneal_cap
        self.annealing = annealing

        if self.annealing:
            self.beta = K.variable(0.0)
        else:
            self.beta = beta

        # Compute total annealing steps
        self.total_anneal_steps = (
            self.number_of_batches
            * (self.n_epochs - int(self.n_epochs * 0.2))
            // self.anneal_cap
        )

        # Dropout parameters
        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder

        # Path to save optimal model
        self.save_path = save_path

        # Create StandardVAE model
        self._create_model()

    def _create_model(self):
        """Build and compile model."""
        # Encoding
        self.x = Input(shape=(self.original_dim,))
        self.x_ = Lambda(lambda x: K.l2_normalize(x, axis=1))(self.x)
        self.dropout_encoder = Dropout(self.drop_encoder)(self.x_)

        self.h = Dense(
            self.intermediate_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            bias_initializer=tf.keras.initializers.truncated_normal(
                stddev=0.001, seed=self.seed
            ),
        )(self.dropout_encoder)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)

        # Sampling
        self.z = Lambda(self._take_sample, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_var]
        )

        # Decoding
        self.h_decoder = Dense(
            self.intermediate_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            bias_initializer=tf.keras.initializers.truncated_normal(
                stddev=0.001, seed=self.seed
            ),
        )
        self.dropout_decoder = Dropout(self.drop_decoder)
        self.x_bar = Dense(self.original_dim)
        self.h_decoded = self.h_decoder(self.z)
        self.h_decoded_ = self.dropout_decoder(self.h_decoded)
        self.x_decoded = self.x_bar(self.h_decoded_)

        # Training
        self.model = Model(self.x, self.x_decoded)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._get_vae_loss,
        )

    def _get_vae_loss(self, x, x_bar):
        """Calculate negative ELBO (NELBO)."""
        log_softmax_var = tf.nn.log_softmax(x_bar)
        self.neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * x, axis=-1))
        a = tf.keras.backend.print_tensor(self.neg_ll)  # noqa: F841
        # calculate positive Kullback–Leibler divergence  divergence term
        kl_loss = K.mean(
            0.5
            * K.sum(
                -1 - self.z_log_var + K.square(self.z_mean) + K.exp(self.z_log_var),
                axis=-1,
            )
        )

        # obtain negative ELBO
        neg_ELBO = self.neg_ll + self.beta * kl_loss

        return neg_ELBO

    def _take_sample(self, args):
        """Sample epsilon ∼ N (0,I) and compute z via reparametrization trick."""

        """Calculate latent vector using the reparametrization trick.
           The idea is that sampling from N (_mean, _var) is s the same as sampling from _mean+ epsilon * _var
           where epsilon ∼ N(0,I)."""
        # _mean and _log_var calculated in encoder
        _mean, _log_var = args

        # epsilon
        epsilon = K.random_normal(
            shape=(K.shape(_mean)[0], self.latent_dim),
            mean=0.0,
            stddev=1.0,
            seed=self.seed,
        )

        return _mean + K.exp(_log_var / 2) * epsilon

    def nn_batch_generator(self, x_train):
        """Used for splitting dataset in batches.

        Args:
            x_train (numpy.ndarray): The click matrix for the train set, with float values.
        """
        # Shuffle the batch
        np.random.seed(self.seed)
        shuffle_index = np.arange(np.shape(x_train)[0])
        np.random.shuffle(shuffle_index)
        x = x_train[shuffle_index, :]
        y = x_train[shuffle_index, :]

        # Iterate until making a full epoch
        counter = 0
        while 1:
            index_batch = shuffle_index[
                self.batch_size * counter : self.batch_size * (counter + 1)
            ]
            # Decompress batch
            x_batch = x[index_batch, :]
            y_batch = y[index_batch, :]
            counter += 1
            yield (np.array(x_batch), np.array(y_batch))

            # Stopping rule
            if counter >= self.number_of_batches:
                counter = 0

    def fit(self, x_train, x_valid, x_val_tr, x_val_te, mapper):
        """Fit model with the train sets and validate on the validation set.

        Args:
            x_train (numpy.ndarray): the click matrix for the train set.
            x_valid (numpy.ndarray): the click matrix for the validation set.
            x_val_tr (numpy.ndarray): the click matrix for the validation set training part.
            x_val_te (numpy.ndarray): the click matrix for the validation set testing part.
            mapper (object): the mapper for converting click matrix to dataframe. It can be AffinityMatrix.
        """
        # initialise LossHistory used for saving loss of validation and train set per epoch
        history = LossHistory()

        # initialise Metrics  used for calculating NDCG@k per epoch
        # and saving the model weights with the highest NDCG@k value
        metrics = Metrics(
            model=self.model,
            val_tr=x_val_tr,
            val_te=x_val_te,
            mapper=mapper,
            k=self.k,
            save_path=self.save_path,
        )

        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001
        )

        if self.annealing is True:
            # initialise AnnealingCallback for annealing process
            anneal = AnnealingCallback(
                self.beta, self.anneal_cap, self.total_anneal_steps
            )

            # fit model
            self.model.fit_generator(
                generator=self.nn_batch_generator(x_train),
                steps_per_epoch=self.number_of_batches,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=[metrics, history, self.reduce_lr, anneal],
                validation_data=(x_valid, x_valid),
            )

            self.ls_beta = anneal.get_data()

        else:
            self.model.fit_generator(
                generator=self.nn_batch_generator(x_train),
                steps_per_epoch=self.number_of_batches,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=[metrics, history, self.reduce_lr],
                validation_data=(x_valid, x_valid),
            )

        # save lists
        self.train_loss = history.losses
        self.val_loss = history.val_losses
        self.val_ndcg = metrics.get_data()

    def get_optimal_beta(self):
        """Returns the value of the optimal beta."""
        if self.annealing is True:
            # find the epoch/index that had the highest NDCG@k value
            index_max_ndcg = np.argmax(self.val_ndcg)

            # using this index find the value that beta had at this epoch
            return self.ls_beta[index_max_ndcg]
        else:
            return self.beta

    def display_metrics(self):
        """Plots:
        1) Loss per epoch both for validation and train set
        2) NDCG@k per epoch of the validation set
        """
        # Plot setup
        plt.figure(figsize=(14, 5))
        sns.set(style="whitegrid")

        # Plot loss on the left graph
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, color="b", linestyle="-", label="Train")
        plt.plot(self.val_loss, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("Loss", size=14)
        plt.legend(loc="upper left")

        # Plot NDCG on the right graph
        plt.subplot(1, 2, 2)
        plt.plot(self.val_ndcg, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("NDCG@k", size=14)
        plt.legend(loc="upper left")

        # Add title
        plt.suptitle("TRAINING AND VALIDATION METRICS HISTORY", size=16)
        plt.tight_layout(pad=2)

    def recommend_k_items(self, x, k, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.
        Obtained probabilities are used as recommendation score.

        Args:
            x (numpy.ndarray, int32): input click matrix.
            k (scalar, int32): the number of items to recommend.
        Returns:
            numpy.ndarray, float: A sparse matrix containing the top_k elements ordered by their score.
        """
        # return optimal model
        self.model.load_weights(self.save_path)

        # obtain scores
        score = self.model.predict(x)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0
        # get the top k items
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]
        # get a copy of the score matrix
        score_c = score.copy()
        # set to zero the k elements
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        # set to zeros all elements other then the k
        top_scores = score - score_c
        return top_scores

    def ndcg_per_epoch(self):
        """Returns the list of NDCG@k at each epoch."""
        return self.val_ndcg
