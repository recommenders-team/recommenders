from __future__ import division

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class Oblique(Manifold):
    """
    Manifold of matrices w/ unit-norm columns.

    Oblique manifold: deals with matrices of size m-by-n such that each column
    has unit 2-norm, i.e., is a point on the unit sphere in R^m. The metric
    is such that the oblique manifold is a Riemannian submanifold of the
    space of m-by-n matrices with the usual trace inner product, i.e., the
    usual metric.
    """

    def __init__(self, m, n):
        self._m = m
        self._n = n

    def __str__(self):
        return "Oblique manifold OB({:d}, {:d})".format(self._m, self._n)

    @property
    def dim(self):
        return (self._m - 1) * self._n

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n)

    def inner(self, X, U, V):
        return float(np.tensordot(U, V))

    def norm(self, X, U):
        return la.norm(U)

    def dist(self, X, Y):
        XY = (X * Y).sum(0)
        XY[XY > 1] = 1
        U = np.arccos(XY)
        return la.norm(U)

    def proj(self, X, H):
        return H - X * ((X * H).sum(0)[np.newaxis, :])

    def ehess2rhess(self, X, egrad, ehess, U):
        PXehess = self.proj(X, ehess)
        return PXehess - U * ((X * egrad).sum(0)[np.newaxis, :])

    def exp(self, X, U):
        norm_U = np.sqrt((U ** 2).sum(0))[np.newaxis, :]

        Y = X * np.cos(norm_U) + U * (np.sin(norm_U) / norm_U)

        # For those columns where the step is too small, use a retraction.
        exclude = np.nonzero(norm_U <= 4.5e-8)[-1]
        Y[:, exclude] = self._normalize_columns(X[:, exclude] + U[:, exclude])

        return Y

    def retr(self, X, U):
        return self._normalize_columns(X + U)

    def log(self, X, Y):
        V = self.proj(X, Y - X)
        dists = np.arccos((X * Y).sum(0))
        norms = np.sqrt((V ** 2).sum(0)).real
        factors = dists / norms
        # For very close points, dists is almost equal to norms, but because
        # they are both almost zero, the division above can return NaN's. To
        # avoid that, we force those ratios to 1.
        factors[dists <= 1e-6] = 1

        return V * factors

    def rand(self):
        return self._normalize_columns(rnd.randn(self._m, self._n))

    def randvec(self, X):
        H = rnd.randn(*X.shape)
        P = self.proj(X, H)
        return P / self.norm(X, P)

    def transp(self, X, Y, U):
        return self.proj(Y, U)

    def pairmean(self, X, Y):
        return self._normalize_columns(X + Y)

    def _normalize_columns(self, X):
        """Return an l2-column-normalized copy of the matrix X."""
        return X / la.norm(X, axis=0)[np.newaxis, :]
