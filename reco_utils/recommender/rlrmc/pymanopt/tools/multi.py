import numpy as np


def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)

    # Approx 5x faster, only supported by numpy version >= 1.6:
    return np.einsum('ijk,ikl->ijl', A, B)


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))


def multiskew(A):
    # Inspired by MATLAB multiskew function by Nicholas Boumal.
    return 0.5 * (A - multitransp(A))


def multieye(k, n):
    # Creates a k x n x n array containing k (n x n) identity matrices.
    return np.tile(np.eye(n), (k, 1, 1))


def multilog(A, pos_def=False):
    if not pos_def:
        raise NotImplementedError

    # Computes the logm of each matrix in an array containing k positive
    # definite matrices. This is much faster than scipy.linalg.logm even
    # for a single matrix. Could potentially be improved further.
    w, v = np.linalg.eigh(A)
    w = np.expand_dims(np.log(w), axis=-1)
    return multiprod(v, w * multitransp(v))


def multiexp(A, sym=False):
    if not sym:
        raise NotImplementedError

    # Compute the expm of each matrix in an array of k symmetric matrices.
    # Sometimes faster than scipy.linalg.expm even for a single matrix.
    w, v = np.linalg.eigh(A)
    w = np.expand_dims(np.exp(w), axis=-1)
    return multiprod(v, w * multitransp(v))
