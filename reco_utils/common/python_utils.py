import os
import numpy as np
from scipy.sparse import coo_matrix


def get_number_processors():
    """Get the number of processors in a CPU.
    Returns:
        int: Number of processors.
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing  # force exception in case mutiprocessing is not installed

        num = multiprocessing.cpu_count()
    return num


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of cooccurrences
    Args:
        cooccurrence (scipy.sparse.csc_matrix): the symmetric matrix of cooccurrences of items
    Returns:
        scipy.sparse.coo_matrix: The matrix of Jaccard similarities between any two items
    """
    coo = cooccurrence.tocoo()
    denom = coo.diagonal()[coo.row] + coo.diagonal()[coo.col] - coo.data
    return coo_matrix((np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=(denom != 0.0)),
                        (coo.row, coo.col)),
                      shape=coo.shape).tocsc()


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
