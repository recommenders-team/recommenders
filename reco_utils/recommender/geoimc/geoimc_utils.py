# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.decomposition import PCA

from reco_utils.dataset.download_utils import maybe_download
from IPython import embed

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


def binarize(a, threshold):
    """Binarize the values.

    Args:
        a (np.ndarray): Input array that needs to be binarized.
        threshold (float): Threshold below which all values are set to 0, else 1.
    """
    return np.where(
        a > threshold,
        1.0,
        0.0
    )


def reduce_dims(matrix, target_dim):
    """Reduce dimensionality of the data using PCA.

    Args:
        matrix (np.ndarray): Matrix of the form (n_sampes, n_features)
        target_dim (uint): Dimension to which n_features should be reduced to.

    """
    model = PCA(n_components=target_dim)
    model.fit(matrix)
    return model.transform(matrix)


def download_geoimc_features(remote_base_url, remote_filenames, dest):
    """A small utility to download features

    Args:
        remote_base_url (url): Base URL at which features are present.
        remote_filenames (Iterator): An iterator (of 2 elements, in general) containing
        the filenames of row, col features at the remote_base_url.
        dest (str): The destination of these downloaded files (Destination dir should already be
        created).
    """
    for _remote_fname in remote_filenames:
        maybe_download(f"{remote_base_url}/{_remote_fname}", f"{dest}/{_remote_fname}")
