# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from scipy.linalg import sqrtm
from numba import njit, jit, prange

from .geoimc_utils import length_normalize
from reco_utils.utils.python_utils import binarize as conv_binary


class PlainScalarProduct(object):
    """
    Module that implements plain scalar product
    as the retrieval criterion
    """

    def __init__(self, X, Y, **kwargs):
        """
        Args:
            X: numpy matrix of shape (users, features)
            Y: numpy matrix of shape (items, features)
        """
        self.X = X
        self.Y = Y

    def sim(self, **kwargs):
        """Calculate the similarity score"""
        sim = self.X.dot(self.Y.T)
        return sim


class Inferer:
    """
    Holds necessary (minimal) information needed for inference
    """

    def __init__(self, method="dot", k=10, transformation=""):
        """Initialize parameters

        Args:
            method (str): The inference method. Currently 'dot'
                (Dot product) is supported.
            k (uint): `k` for 'topk' transformation.
            transformation (str): Transform the inferred values into a
                different scale. Currently 'mean' (Binarize the values
                using mean of inferred matrix as the threshold), 'topk'
                (Pick Top-K inferred values per row and assign them 1,
                setting rest of them to 0), '' (No transformation) are
                supported.
        """
        self.method = self._get_method(method)
        self.k = k
        self.transformation = transformation

    def _get_method(self, k):
        """Get the inferer method

        Args:
            k (str): The inferer name

        Returns:
            class: A class object implementing the inferer 'k'
        """
        if k == "dot":
            method = PlainScalarProduct
        else:
            raise ValueError(f"{k} is unknown.")
        return method

    def infer(self, dataPtr, W, **kwargs):
        """Main inference method

        Args:
            dataPtr (DataPtr): An object containing the X, Z features needed for inference
            W (iterable): An iterable containing the U, B, V parametrized matrices.
        """

        if isinstance(dataPtr, list):
            a = dataPtr[0]
            b = dataPtr[1]
        else:
            a = dataPtr.get_entity("row").dot(W[0]).dot(sqrtm(W[1]))
            b = dataPtr.get_entity("col").dot(W[2]).dot(sqrtm(W[1]))

        sim_score = self.method(a, b).sim(**kwargs)

        if self.transformation == "mean":
            prediction = conv_binary(sim_score, sim_score.mean())
        elif self.transformation == "topk":
            masked_sim_score = sim_score.copy()

            for i in range(sim_score.shape[0]):
                topKidx = np.argpartition(masked_sim_score[i], -self.k)[-self.k :]
                mask = np.ones(sim_score[i].size, dtype=bool)
                mask[topKidx] = False

                masked_sim_score[i][topKidx] = 1
                masked_sim_score[i][mask] = 0
            prediction = masked_sim_score
        else:
            prediction = sim_score

        return prediction
