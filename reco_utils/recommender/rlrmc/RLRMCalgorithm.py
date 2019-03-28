import random
import numpy as np
import pandas as pd
import warnings

randomize_seed = 10 # fixing a random seed for reproducibility of results
random.seed(randomize_seed)
from numpy.random import seed
seed(randomize_seed)

import sys
# Path to the folder where Pymanopt's code is residing. 
pymanopt_path = "..//third_party_tools" 
# The Pymanopt code has been downloaded via: pip install pymanopt  
# Online code of Pymanopt: https://github.com/pymanopt/pymanopt
# Online license link: https://github.com/pymanopt/pymanopt/blob/master/LICENSE
# Pymanopt is licensed under the BSD 3-Clause "New" or "Revised" License - 
# A permissive license similar to the BSD 2-Clause License, but with a 
# 3rd clause that prohibits others from using the name of the project or its 
# contributors to promote derived products without written consent.
# The offline license file of pymanopt can be found in the directory 
# "..\\third_party_tools\pymanopt"
sys.path.insert(0, pymanopt_path)

from pymanopt import Problem
from pymanopt.solvers import ConjugateGradientMS # Modified Conjugate Gradient 
from pymanopt.solvers.linesearch import LineSearchBackTracking
from pymanopt.manifolds import Stiefel, PositiveDefinite, Product
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import time 
from numba import jit, njit, prange


class RLRMCalgorithm(object):
    """
    classdocs
    """

    def __init__(
        self,
        rank,
        C,
        model_param,
        initialize_flag = 'random',
        max_time = 1000,
        maxiter = 100,
        seed = 42,
    ):
        """
        Constructor
        """
        self.model_param = model_param
        self.train_mean = model_param.get('train_mean')
        self.initialize_flag = initialize_flag
        self.rank = rank
        self.C = C
        self.max_time = max_time
        self.maxiter = maxiter

    def _init_train(self,entries_train_csr):
        print("Hyper-parameters of the algorithm")
        print("Rank: %i, Regularization parameter: %e" % (self.rank, self.C))
        # Initialization # starting point on the manifold
        initialize_dict = {'random': 0, 'svd': 1}
        # t1 = time.time()
        if initialize_dict.get(self.initialize_flag)==0:#rndom
            W0=None
            # self.init_time = 0.0
        elif initialize_dict.get(self.initialize_flag)==1:#svd
            U0, B0, V0 = svds(entries_train_csr, k=self.rank)
            W0 = [U0, V0.T, np.diag(B0)]
            # self.init_time = time.time()-t1
            # print("\nSVD-based initialization time: %.2f sec.\n" % (init_time))
        else:#default option when given incorrect option
            print("Initialization flag not recognized. Setting it to random (default).")
            W0=None
            # self.init_time = 0.0
        return W0

    def fit(self, entries_train_csr, entries_test_csr=None, verbosity = 0, iterwise_rmse = False):
    	# train data
        # self.entries_train_csr = entries_train_csr
        # self.entries_train_csr_data = entries_train_csr.data
        # self.entries_train_csr_indices = entries_train_csr.indices
        # self.entries_train_csr_indptr = entries_train_csr.indptr

        # initialize the model
        W0 = self._init_train(entries_train_csr)

        # test data
        # self.entries_test_csr = entries_test_csr
        # self.entries_test_csr_data = entries_test_csr.data
        # self.entries_test_csr_indices = entries_test_csr.indices
        # self.entries_test_csr_indptr = entries_test_csr.indptr

        # global variable residual
        residual_global = np.zeros(entries_train_csr.data.shape,dtype=np.float64)

        ###################Riemannian first-order algorithm######################
        # first-order algorithm
        solver = ConjugateGradientMS(maxtime=self.max_time, 
            maxiter=self.maxiter, linesearch=LineSearchBackTracking())#, logverbosity=2)
        # construction of manifold
        manifold = Product([
            Stiefel(self.model_param.get('num_row'), self.rank), 
            Stiefel(self.model_param.get('num_col'), self.rank), 
            PositiveDefinite(self.rank)
            ])
        problem = Problem(
            manifold=manifold, 
            cost=lambda x: self.cost(x,entries_train_csr.data,entries_train_csr.indices,entries_train_csr.indptr,residual_global), 
            egrad=lambda z: self.egrad(z,entries_train_csr.indices,entries_train_csr.indptr,residual_global), 
            verbosity=verbosity)
        
        if iterwise_rmse:
            if (entries_test_csr is None):
                Wopt, self.stats = solver.solve(problem, x=W0,
                    compute_stats=lambda x,y,z: self.my_stats(x,y,z,residual_global))
        if (entries_test_csr is None) or (not iterwise_test_rmse):
            Wopt, self.stats = solver.solve(problem, x=W0)
        else:
            residual_test_global = (np.zeros(entries_test_csr.data.shape,dtype=np.float64))
            Wopt, self.stats = solver.solve(problem, x=W0, 
                compute_stats=lambda x,y,z: self.my_stats(x,y,z,residual_global,entries_test_csr.data,entries_test_csr.indices,entries_test_csr.indptr,residual_test_global))
        self.L = np.dot(Wopt[0], Wopt[2])
        self.R = Wopt[1].T
        

    # computes residual_global = a*b - cd at given indices in csr_matrix format
    @staticmethod
    @njit(nogil=True,parallel=True)
    def computeLoss_csrmatrix(a,b,cd,indices,indptr,residual_global):
        N = a.shape[0]
        M = a.shape[1]
        for i in prange(N):
            for j in prange(indptr[i],indptr[i+1]):
                num = 0.0
                for k in range(M):
                    num += a[i,k]*b[k,indices[j]]
                residual_global[j] = num - cd[j]
        return residual_global

    # computes user-defined statistics per iteration
    def my_stats(self,weights,given_stats,stats,residual_global,entries_test_csr_data=None,
        entries_test_csr_indices=None,entries_test_csr_indptr=None,residual_test_global=None):
        iter = given_stats[0]
        cost = given_stats[1]
        gradnorm = given_stats[2]
        time_iter = given_stats[3]
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        train_mse = np.mean(residual_global ** 2)
        train_rmse = sqrt(train_mse)
        # Prediction
        if entries_test_csr_data is not None:
            RLRMCalgorithm.computeLoss_csrmatrix(
                U1_dot_B,
                U2.T,
                entries_test_csr_data,
                entries_test_csr_indices,
                entries_test_csr_indptr,
                residual_test_global)
            test_mse = np.mean(residual_test_global ** 2)
            test_rmse = sqrt(test_mse)
            print(
                'Train RMSE: %.4f, Test RMSE: %.4f, Total time: %.2f' 
                % (train_rmse, test_rmse,time_iter))
        stats.setdefault("iteration", []).append(iter)
        stats.setdefault("time", []).append(time_iter)
        stats.setdefault("objective", []).append(cost)
        stats.setdefault("gradnorm", []).append(gradnorm)
        stats.setdefault("trainRMSE", []).append(train_rmse)
        stats.setdefault("testRMSE", []).append(test_rmse)
        return

    # computes the objective function at a given point
    def cost(self, weights,entries_train_csr_data,entries_train_csr_indices,entries_train_csr_indptr,residual_global):
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        RLRMCalgorithm.computeLoss_csrmatrix(
            U1_dot_B,
            U2.T,
            entries_train_csr_data,
            entries_train_csr_indices,
            entries_train_csr_indptr,
            residual_global)
        objective = (
            0.5*np.sum((residual_global)**2) + 0.5*self.C*np.sum(B**2))
        return objective

    # computes the gradient of the objective function at a given point
    def egrad(self,weights,entries_train_csr_indices,entries_train_csr_indptr,residual_global):
        U1 = weights[0]
        U2 = weights[1]
        B = weights[2]
        U1_dot_B = np.dot(U1, B)
        residual_global_csr = csr_matrix((residual_global, 
            entries_train_csr_indices,entries_train_csr_indptr),shape=(U1.shape[0],U2.shape[0]))
        residual_global_csr_dot_U2 = residual_global_csr.dot(U2)
        gradU1 = np.dot(residual_global_csr_dot_U2,B)
        gradB_asymm = np.dot(U1.T,residual_global_csr_dot_U2) + self.C*B
        gradB = (gradB_asymm + gradB_asymm.T)/2.0
        gradU2 = residual_global_csr.T.dot(U1_dot_B)
        return [gradU1, gradU2, gradB]

