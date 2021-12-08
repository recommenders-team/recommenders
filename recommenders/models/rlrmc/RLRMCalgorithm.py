# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import logging

from pymanopt import Problem
from recommenders.models.rlrmc.conjugate_gradient_ms import ConjugateGradientMS
from pymanopt.solvers.linesearch import LineSearchBackTracking
from pymanopt.manifolds import Stiefel, SymmetricPositiveDefinite, Product
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from numba import njit, prange


logger = logging.getLogger(__name__)


class RLRMCalgorithm(object):
    """RLRMC algorithm implementation."""

    def __init__(
        self,
        rank,
        C,
        model_param,
        initialize_flag="random",
        max_time=1000,
        maxiter=100,
        seed=42,
    ):
        """Initialize parameters.

        Args:
            rank (int): rank of the final model. Should be a positive integer.
            C (float): regularization parameter. Should be a positive real number.
            model_param (dict): contains model parameters such as number of rows & columns of the matrix as well as
                the mean rating in the training dataset.
            initialize_flag (str): flag to set the initialization step of the algorithm. Current options are 'random'
                (which is random initilization) and 'svd' (which is a singular value decomposition based initilization).
            max_time (int): maximum time (in seconds), for which the algorithm is allowed to execute.
            maxiter (int): maximum number of iterations, for which the algorithm is allowed to execute.
        """
        self.model_param = model_param
        self.train_mean = model_param.get("train_mean")
        self.initialize_flag = initialize_flag
        self.rank = rank
        self.C = C
        self.max_time = max_time
        self.maxiter = maxiter

    def _init_train(self, entries_train_csr):
        logger.info("Hyper-parameters of the algorithm")
        logger.info("Rank: %i, Regularization parameter: %e" % (self.rank, self.C))
        # Initialization # starting point on the manifold
        if self.initialize_flag == "random":  # rndom
            W0 = None
        elif self.initialize_flag == "svd":  # svd
            U0, B0, V0 = svds(entries_train_csr, k=self.rank)
            W0 = [U0, V0.T, np.diag(B0)]
        else:  # default option when given incorrect option
            logger.warning(
                "Initialization flag not recognized. Setting it to random (default)."
            )
            W0 = None
        return W0

    def fit_and_evaluate(self, RLRMCdata, verbosity=0):
        """Main fit and evalute method for RLRMC. In addition to fitting the model, it also computes the per
        iteration statistics in train (and validation) datasets.

        Args:
            RLRMCdata (RLRMCdataset): the RLRMCdataset object.
            verbosity (int): verbosity of Pymanopt. Possible values are 0 (least verbose), 1, or 2 (most verbose).
        """
        # it calls fit method with appropriate arguments
        self.fit(RLRMCdata, verbosity, True)

    def fit(self, RLRMCdata, verbosity=0, _evaluate=False):
        """The underlying fit method for RLRMC

        Args:
            RLRMCdata (RLRMCdataset): the RLRMCdataset object.
            verbosity (int): verbosity of Pymanopt. Possible values are 0 (least verbose), 1, or 2 (most verbose).
            _evaluate (bool): flag to compute the per iteration statistics in train (and validation) datasets.
        """
        # initialize the model
        W0 = self._init_train(RLRMCdata.train)
        self.user2id = RLRMCdata.user2id
        self.item2id = RLRMCdata.item2id
        self.id2user = RLRMCdata.id2user
        self.id2item = RLRMCdata.id2item

        # residual variable
        residual_global = np.zeros(RLRMCdata.train.data.shape, dtype=np.float64)

        ####################################
        # Riemannian first-order algorithm #
        ####################################

        solver = ConjugateGradientMS(
            maxtime=self.max_time,
            maxiter=self.maxiter,
            linesearch=LineSearchBackTracking(),
        )  # , logverbosity=2)
        # construction of manifold
        manifold = Product(
            [
                Stiefel(self.model_param.get("num_row"), self.rank),
                Stiefel(self.model_param.get("num_col"), self.rank),
                SymmetricPositiveDefinite(self.rank),
            ]
        )
        problem = Problem(
            manifold=manifold,
            cost=lambda x: self._cost(
                x,
                RLRMCdata.train.data,
                RLRMCdata.train.indices,
                RLRMCdata.train.indptr,
                residual_global,
            ),
            egrad=lambda z: self._egrad(
                z, RLRMCdata.train.indices, RLRMCdata.train.indptr, residual_global
            ),
            verbosity=verbosity,
        )

        if _evaluate:
            residual_validation_global = np.zeros(
                RLRMCdata.validation.data.shape, dtype=np.float64
            )
            Wopt, self.stats = solver.solve(
                problem,
                x=W0,
                compute_stats=lambda x, y, z: self._my_stats(
                    x,
                    y,
                    z,
                    residual_global,
                    RLRMCdata.validation.data,
                    RLRMCdata.validation.indices,
                    RLRMCdata.validation.indptr,
                    residual_validation_global,
                ),
            )
        else:
            Wopt, self.stats = solver.solve(problem, x=W0)
        self.L = np.dot(Wopt[0], Wopt[2])
        self.R = Wopt[1]

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _computeLoss_csrmatrix(a, b, cd, indices, indptr, residual_global):
        """computes residual_global = a*b - cd at given indices in csr_matrix format"""
        N = a.shape[0]
        M = a.shape[1]
        for i in prange(N):
            for j in prange(indptr[i], indptr[i + 1]):
                num = 0.0
                for k in range(M):
                    num += a[i, k] * b[k, indices[j]]
                residual_global[j] = num - cd[j]
        return residual_global

    # computes user-defined statistics per iteration
    def _my_stats(
        self,
        weights,
        given_stats,
        stats,
        residual_global,
        entries_validation_csr_data=None,
        entries_validation_csr_indices=None,
        entries_validation_csr_indptr=None,
        residual_validation_global=None,
    ):
        iteration = given_stats[0]
        cost = given_stats[1]
        gradnorm = given_stats[2]
        time_iter = given_stats[3]
        stats.setdefault("iteration", []).append(iteration)
        stats.setdefault("time", []).append(time_iter)
        stats.setdefault("objective", []).append(cost)
        stats.setdefault("gradnorm", []).append(gradnorm)
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        train_mse = np.mean(residual_global ** 2)
        train_rmse = sqrt(train_mse)
        stats.setdefault("trainRMSE", []).append(train_rmse)
        # Prediction
        if entries_validation_csr_data is not None:
            RLRMCalgorithm._computeLoss_csrmatrix(
                U1_dot_B,
                U2.T,
                entries_validation_csr_data,
                entries_validation_csr_indices,
                entries_validation_csr_indptr,
                residual_validation_global,
            )
            validation_mse = np.mean(residual_validation_global ** 2)
            validation_rmse = sqrt(validation_mse)
            stats.setdefault("validationRMSE", []).append(validation_rmse)
            logger.info(
                "Train RMSE: %.4f, Validation RMSE: %.4f, Total time: %.2f"
                % (train_rmse, validation_rmse, time_iter)
            )
        else:
            logger.info("Train RMSE: %.4f, Total time: %.2f" % (train_rmse, time_iter))
        return

    # computes the objective function at a given point
    def _cost(
        self,
        weights,
        entries_train_csr_data,
        entries_train_csr_indices,
        entries_train_csr_indptr,
        residual_global,
    ):
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        RLRMCalgorithm._computeLoss_csrmatrix(
            U1_dot_B,
            U2.T,
            entries_train_csr_data,
            entries_train_csr_indices,
            entries_train_csr_indptr,
            residual_global,
        )
        objective = 0.5 * np.sum((residual_global) ** 2) + 0.5 * self.C * np.sum(B ** 2)
        return objective

    # computes the gradient of the objective function at a given point
    def _egrad(
        self,
        weights,
        entries_train_csr_indices,
        entries_train_csr_indptr,
        residual_global,
    ):
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        residual_global_csr = csr_matrix(
            (residual_global, entries_train_csr_indices, entries_train_csr_indptr),
            shape=(U1.shape[0], U2.shape[0]),
        )
        residual_global_csr_dot_U2 = residual_global_csr.dot(U2)
        gradU1 = np.dot(residual_global_csr_dot_U2, B)
        gradB_asymm = np.dot(U1.T, residual_global_csr_dot_U2) + self.C * B
        gradB = (gradB_asymm + gradB_asymm.T) / 2.0
        gradU2 = residual_global_csr.T.dot(U1_dot_B)
        return [gradU1, gradU2, gradB]

    def predict(self, user_input, item_input, low_memory=False):
        """Predict function of this trained model

        Args:
            user_input ( list or element of list ): userID or userID list
            item_input ( list or element of list ): itemID or itemID list

        Returns:
            list or float: list of predicted rating or predicted rating score.
        """
        # index converting
        user_input = np.array([self.user2id[x] for x in user_input])  # rows
        item_input = np.array([self.item2id[x] for x in item_input])  # columns
        num_test = user_input.shape[0]
        if num_test != item_input.shape[0]:
            print("ERROR! Dimension mismatch in test data.")
            return None
        output = np.empty(item_input.shape, dtype=np.float64)
        output.fill(-self.train_mean)
        L = self.L
        R = self.R
        if low_memory:
            # for-loop
            for i in np.arange(num_test):
                output[i] += np.dot(L[user_input[i], :], R[item_input[i], :])
        else:
            # matrix multiplication
            d = self.model_param.get("num_row")
            T = self.model_param.get("num_col")
            test = csr_matrix((output, (user_input, item_input)), shape=(d, T))
            RLRMCalgorithm._computeLoss_csrmatrix(
                L, R.T, test.data, test.indices, test.indptr, output
            )
            lin_index_org = np.ravel_multi_index(
                (user_input, item_input), dims=(d, T), mode="raise", order="C"
            )
            idx1 = np.argsort(lin_index_org)
            idx2 = np.argsort(idx1)
            output = output[idx2]
        return output
