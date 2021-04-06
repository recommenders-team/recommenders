# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.decomposition import PCA

from reco_utils.dataset.download_utils import maybe_download


def length_normalize(matrix):
    """Length normalize the matrix

    Args:
        matrix (np.ndarray): Input matrix that needs to be normalized

    Returns:
        Normalized matrix
    """
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def mean_center(matrix):
    """Performs mean centering across axis 0

    Args:
        matrix (np.ndarray): Input matrix that needs to be mean centered
    """
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def reduce_dims(matrix, target_dim):
    """Reduce dimensionality of the data using PCA.

    Args:
        matrix (np.ndarray): Matrix of the form (n_sampes, n_features)
        target_dim (uint): Dimension to which n_features should be reduced to.

    """
    model = PCA(n_components=target_dim)
    model.fit(matrix)
    return model.transform(matrix)
