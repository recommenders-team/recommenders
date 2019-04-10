from __future__ import division

import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class Sphere(Manifold):
    """
    Manifold of shape n1 x n2 x ... x nk tensors with unit 2-norm. The
    metric is such that the sphere is a Riemannian submanifold of Euclidean
    space. This implementation is based on spherefactory.m from the Manopt
    MATLAB package.

    Examples:
    The 'usual' sphere S^2, the set of points lying on the surface of a ball in
    3D space:
    sphere = Sphere(3)
    """

    def __init__(self, *shape):
        self._shape = shape
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Sphere manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = "Sphere manifold of {}x{} matrices".format(*shape)
        else:
            self._name = "Sphere manifold of shape " + str(shape) + " tensors"

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape) - 1

    @property
    def typicaldist(self):
        return np.pi

    def inner(self, X, U, V):
        return float(np.tensordot(U, V, axes=U.ndim))

    def norm(self, X, U):
        return la.norm(U)

    def dist(self, U, V):
        # Make sure inner product is between -1 and 1
        inner = max(min(self.inner(None, U, V), 1), -1)
        return np.arccos(inner)

    def proj(self, X, H):
        return H - self.inner(None, X, H) * X

    def ehess2rhess(self, X, egrad, ehess, U):
        return self.proj(X, ehess) - self.inner(None, X, egrad) * U

    def exp(self, X, U):
        norm_U = self.norm(None, U)
        # Check that norm_U isn't too tiny. If very small then
        # sin(norm_U) / norm_U ~= 1 and retr is extremely close to exp.
        if norm_U > 1e-3:
            return X * np.cos(norm_U) + U * np.sin(norm_U) / norm_U
        else:
            return self.retr(X, U)

    def retr(self, X, U):
        Y = X + U
        return self._normalize(Y)

    def log(self, X, Y):
        P = self.proj(X, Y - X)
        dist = self.dist(X, Y)
        # If the two points are "far apart", correct the norm.
        if dist > 1e-6:
            P *= dist / self.norm(None, P)
        return P

    def rand(self):
        Y = rnd.randn(*self._shape)
        return self._normalize(Y)

    def randvec(self, X):
        H = rnd.randn(*self._shape)
        P = self.proj(X, H)
        return self._normalize(P)

    def transp(self, X, Y, U):
        return self.proj(Y, U)

    def pairmean(self, X, Y):
        return self._normalize(X + Y)

    def _normalize(self, X):
        """
        Return a Frobenius-normalized version of the point X in the ambient
        space.
        """
        return X / self.norm(None, X)


class SphereSubspaceIntersection(Sphere):
    """
    Manifold of n-dimensional unit 2-norm vectors intersecting the
    r-dimensional subspace of R^n spanned by the columns of the matrix U. This
    implementation is based on spheresubspacefactory.m from the Manopt MATLAB
    package.
    """

    def __init__(self, n, U=None):
        super(SphereSubspaceIntersection, self).__init__(n)

        self._n = n
        if U is None:
            self._subspace_projector = np.eye(n)
            self._subspace_dimension = n
            # The name is defined in the base class.
        else:
            if U.shape[0] != n:
                raise ValueError(
                    "Number of rows in U does not match dimension of the "
                    "ambient space.")
            self._configure_manifold(U)

        if self.dim == 0:
            warnings.warn("Manifold only consists of isolated points when "
                          "subspace is 1-dimensional.")

    def _configure_manifold(self, U):
        Q, _ = la.qr(U)
        self._subspace_projector = Q.dot(Q.T)
        self._subspace_dimension = la.matrix_rank(self._subspace_projector)
        self._name = ("Sphere manifold of {}-dimensional vectors intersecting "
                      "a {}-dimensional subspace".format(
                          self._n, self._subspace_dimension))

    @property
    def dim(self):
        return self._subspace_dimension - 1

    def proj(self, X, H):
        Y = super(SphereSubspaceIntersection, self).proj(X, H)
        return self._subspace_projector.dot(Y)

    def rand(self):
        X = super(SphereSubspaceIntersection, self).rand()
        return self._normalize(self._subspace_projector.dot(X))

    def randvec(self, X):
        Y = super(SphereSubspaceIntersection, self).randvec(X)
        return self._normalize(self._subspace_projector.dot(Y))


class SphereSubspaceComplementIntersection(SphereSubspaceIntersection):
    """
    Manifold of n-dimensional unit 2-norm vectors which are orthogonal to the
    r-dimensional subspace of R^n spanned by columns of the matrix U. This
    implementation is based on spheresubspacefactory.m from the Manopt MATLAB
    package.
    """

    def _configure_manifold(self, U):
        Q, _ = la.qr(U)
        self._subspace_projector = np.eye(self._n) - Q.dot(Q.T)
        self._subspace_dimension = la.matrix_rank(self._subspace_projector)
        self._name = ("Sphere manifold of {}-dimensional vectors orthogonal "
                      "to a {}-dimensional subspace".format(
                          self._n, self._subspace_dimension))
