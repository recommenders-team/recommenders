import numpy as np
from sklearn.decomposition import PCA

from numba import jit
from IPython import embed

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def binarize(a, threshold):
    return np.where(
        a > threshold,
        1.0,
        0.0
    )


def reduce_dims(matrix, target_dim):
    model = PCA(n_components=target_dim)
    model.fit(matrix)
    return model.transform(matrix)
