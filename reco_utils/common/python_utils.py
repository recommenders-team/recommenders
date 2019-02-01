import numpy as np
from scipy.sparse import coo_matrix


def exponential_decay(value, max_val, half_life):
    """Compute decay factor for a given value based on an exponential decay
    Args:
        value (numeric): value to calculate decay factor
        max_val (numeric): value at which decay factor will be 1
        half_life (numeric): value at which decay factor will be 0.5
    Returns:
        (float): decay factor
    """
    return np.exp(-np.log(2) * (max_val - value) / half_life)


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of cooccurrences
    Args:
        cooccurrence (scipy.sparse.csc_matrix): the symmetric matrix of cooccurrences of items
    Returns:
        scipy.sparse.coo_matrix: The matrix of Jaccard similarities between any two items
    """
    coo = cooccurrence.tocoo()
    denom = coo.diagonal()[coo.row] + coo.diagonal()[coo.col] - coo.data
    return coo_matrix((np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=(denom != 0.0)), (coo.row, coo.col)), shape=coo.shape).tocsc()


def lift(cooccurrence):
    """Helper method to calculate the Lift of a matrix of cooccurrences
    Args:
        cooccurrence (scipy.sparse.csc_matrix): the symmetric matrix of cooccurrences of items
    Returns:
        scipy.sparse.coo_matrix: The matrix of Lifts between any two items
    """
    coo = cooccurrence.tocoo()
    denom = coo.diagonal()[coo.row] * coo.diagonal()[coo.col]
    return coo_matrix((np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=(denom != 0.0)),
                        (coo.row, coo.col)),
                       shape=coo.shape).tocsc()
