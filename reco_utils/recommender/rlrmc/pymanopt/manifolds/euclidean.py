from __future__ import division

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multisym, multiskew


class Euclidean(Manifold):
    """
    Euclidean manifold of shape n1 x n2 x ... x nk tensors. Useful for
    unconstrained optimization problems or for unconstrained hyperparameters,
    as part of a product manifold.

    Examples:
    Create a manifold of vectors of length n:
    manifold = Euclidean(n)

    Create a manifold of m x n matrices:
    manifold = Euclidean(m, n)
    """

    def __init__(self, *shape):
        self._shape = shape
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Euclidean manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = ("Euclidean manifold of {}x{} "
                          "matrices").format(*shape)
        else:
            self._name = ("Euclidean manifold of shape " + str(shape) +
                          " tensors")

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, X, G, H):
        return float(np.tensordot(G, H, axes=G.ndim))

    def norm(self, X, G):
        return la.norm(G)

    def dist(self, X, Y):
        return la.norm(X-Y)

    def proj(self, X, U):
        return U

    def egrad2rgrad(self, X, U):
        return U

    def ehess2rhess(self, X, egrad, ehess, H):
        return ehess

    def exp(self, X, U):
        return X+U

    retr = exp

    def log(self, X, Y):
        return Y-X

    def rand(self):
        return rnd.randn(*self._shape)

    def randvec(self, X):
        Y = self.rand()
        return Y / self.norm(X, Y)

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return .5*(X+Y)


class Symmetric(Euclidean):
    """
    Manifold of n x n symmetric matrices, as a Riemannian submanifold of
    Euclidean space.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if k == 1:
            self._shape = (n, n)
            self._name = ("Manifold of {} x {} symmetric matrices."
                          ).format(n, n)
        elif k > 1:
            self._shape = (k, n, n)
            self._name = ("Product manifold of {} ({} x {}) symmetric "
                          "matrices.").format(k, n, n)
        else:
            raise ValueError("k must be an integer no less than 1.")

        self._dim = 0.5 * k * n * (n + 1)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def proj(self, X, U):
        return multisym(U)

    def egrad2rgrad(self, X, U):
        return multisym(U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return multisym(ehess)

    def rand(self):
        return multisym(rnd.randn(*self._shape))

    def randvec(self, X):
        Y = self.rand()
        return multisym(Y / self.norm(X, Y))


class SkewSymmetric(Euclidean):
    """
    The Euclidean space of n-by-n skew-symmetric matrices.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if k == 1:
            self._shape = (n, n)
            self._name = ("Manifold of {} x {} skew-symmetric matrices."
                          ).format(n, n)
        elif k > 1:
            self._shape = (k, n, n)
            self._name = ("Product manifold of {} ({} x {}) skew-symmetric "
                          "matrices.").format(k, n, n)
        else:
            raise ValueError("k must be an integer no less than 1.")

        self._dim = .5 * k * n * (n - 1)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def proj(self, X, U):
        return multiskew(U)

    def egrad2rgrad(self, X, U):
        return multiskew(U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return multiskew(ehess)

    def rand(self):
        return multiskew(rnd.randn(*self._shape))

    def randvec(self, X):
        G = self.rand()
        return multiskew(G / self.norm(X, G))
