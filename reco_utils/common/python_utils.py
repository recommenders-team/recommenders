import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of cooccurrences
    cooccurrence: scipy.sparse.csc_matrix
    """
    coo = cooccurrence.tocoo()
    denom = coo.row + coo.col - coo.data
    return coo_matrix((np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=(denom != 0.0)),
                       (coo.row, coo.col)),
                      shape=coo.shape).tocsc()


def lift(cooccurrence):
    """Helper method to calculate the Lift of a matrix of cooccurrences
    cooccurrence: scipy.sparse.csc_matrix
    """
    coo = cooccurrence.tocoo()
    denom = coo.row * coo.col
    return coo_matrix((np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=(denom != 0.0)),
                        (coo.row, coo.col)),
                       shape=coo.shape).tocsc()
