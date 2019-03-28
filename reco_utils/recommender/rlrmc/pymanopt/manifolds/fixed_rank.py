"""
Module containing manifolds of fixed rank matrices.
"""

from __future__ import division

import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds import Stiefel
from pymanopt.tools import ndarraySequenceMixin


class FixedRankEmbedded(Manifold):
    """
    Note: Currently not compatible with the second order TrustRegions solver.
    Should be fixed soon.

    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    For efficiency purposes, Pymanopt does not represent points on this
    manifold explicitly using m x n matrices, but instead implicitly using
    a truncated singular value decomposition. Specifically, a point is
    represented by a tuple (u, s, vt) of three numpy arrays. The arrays u,
    s and vt have shapes (m, k), (k,) and (k, n) respectively, and the low
    rank matrix which they represent can be recovered by the matrix product
    u * diag(s) * vt.

    For example, to optimize over the space of 5 by 4 matrices with rank 3,
    we would need to
    >>> import pymanopt.manifolds
    >>> manifold = pymanopt.manifolds.FixedRankEmbedded(5, 4, 3)

    Then the shapes will be as follows:
    >>> x = manifold.rand()
    >>> x[0].shape
    (5, 3)
    >>> x[1].shape
    (3,)
    >>> x[2].shape
    (3, 4)

    and the full matrix can be recovered using the matrix product
    x[0] * diag(x[1]) * x[2]:
    >>> import numpy as np
    >>> X = x[0].dot(np.diag(x[1])).dot(x[2])

    Tangent vectors are represented as a tuple (Up, M Vp). The matrices Up
    (mxk) and Vp (nxk) obey Up'*U = 0 and Vp'*V = 0.
    The matrix M (kxk) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of mxn matrices:
      Z = U*M*V' + Up*V' + U*Vp'
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as mxn matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U*S*V'. There are no restrictions on what
    U, S and V are, as long as their product as indicated yields a real, mxn
    matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(mxn) equipped with the usual trace (Frobenius) inner product.


    Please cite the Pymanopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }

    This file is based on fixedrankembeddedfactory from Manopt: www.manopt.org.
    Ported by: Jamie Townsend, Sebastian Weichwald
    Original author: Nicolas Boumal, Dec. 30, 2012.
    """

    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k

        self._name = ("Manifold of {m}-by-{n} matrices with rank {k} and "
                      "embedded geometry".format(m=m, n=n, k=k))

        self._stiefel_m = Stiefel(m, k)
        self._stiefel_n = Stiefel(n, k)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return self.dim

    def dist(self, X, Y):
        raise NotImplementedError

    def inner(self, X, G, H):
        return np.sum(np.tensordot(a, b) for (a, b) in zip(G, H))

    def _apply_ambient(self, Z, W):
        """
        For a given ambient vector Z, given as a tuple (U, S, V) such that
        Z = U*S*V', applies it to a matrix W to calculate the matrix product
        ZW.
        """
        if isinstance(Z, tuple):
            return np.dot(Z[0], np.dot(Z[1], np.dot(Z[2].T, W)))
        else:
            return np.dot(Z, W)

    def _apply_ambient_transpose(self, Z, W):
        """
        Same as apply_ambient, but applies Z' to W.
        """
        if isinstance(Z, tuple):
            return np.dot(Z[2], np.dot(Z[1], np.dot(Z[0].T, W)))
        else:
            return np.dot(Z.T, W)

    def proj(self, X, Z):
        """
        Note that Z must either be an m x n matrix from the ambient space, or
        else a tuple (Uz, Sz, Vz), where Uz * Sz * Vz is in the ambient space
        (of low-rank matrices).

        This function then returns a tangent vector parameterized as
        (Up, M, Vp), as described in the class docstring.
        """
        ZV = self._apply_ambient(Z, X[2].T)
        UtZV = np.dot(X[0].T, ZV)
        ZtU = self._apply_ambient_transpose(Z, X[0])

        Up = ZV - np.dot(X[0], UtZV)
        M = UtZV
        Vp = ZtU - np.dot(X[2].T, UtZV.T)

        return _FixedRankTangentVector((Up, M, Vp))

    def egrad2rgrad(self, x, egrad):
        """
        Assuming that the cost function being optimized has been defined
        in terms of the low-rank singular value decomposition of X, the
        gradient returned by the autodiff backends will have three components
        and will be in the form of a tuple egrad = (df/dU, df/dS, df/dV).

        This function correctly maps a gradient of this form into the tangent
        space. See https://j-towns.github.io/papers/svd-derivative.pdf for a
        derivation.
        """
        utdu = np.dot(x[0].T, egrad[0])
        uutdu = np.dot(x[0], utdu)
        Up = (egrad[0] - uutdu) / x[1]

        vtdv = np.dot(x[2], egrad[2].T)
        vvtdv = np.dot(x[2].T, vtdv)
        Vp = (egrad[2].T - vvtdv) / x[1]

        i = np.eye(self._k)
        f = 1 / (x[1][np.newaxis, :]**2 - x[1][:, np.newaxis]**2 + i)

        M = (f * (utdu - utdu.T) * x[1] +
             x[1][:, np.newaxis] * f * (vtdv - vtdv.T) + np.diag(egrad[1]))

        return _FixedRankTangentVector((Up, M, Vp))

    def ehess2rhess(self, X, egrad, ehess, H):
        raise NotImplementedError

    # This retraction is second order, following general results from
    # Absil, Malick, "Projection-like retractions on matrix manifolds",
    # SIAM J. Optim., 22 (2012), pp. 135-158.
    def retr(self, X, Z):
        Qu, Ru = np.linalg.qr(Z[0])
        Qv, Rv = np.linalg.qr(Z[2])

        T = np.vstack((np.hstack((np.diag(X[1]) + Z[1], Rv.T)),
                      np.hstack((Ru, np.zeros((self._k, self._k))))))

        # Numpy svd outputs St as a 1d vector, not a matrix.
        Ut, St, Vt = np.linalg.svd(T, full_matrices=False)

        # Transpose because numpy outputs it the wrong way.
        Vt = Vt.T

        U = np.dot(np.hstack((X[0], Qu)), Ut[:, :self._k])
        V = np.dot(np.hstack((X[2].T, Qv)), Vt[:, :self._k])
        S = St[:self._k] + np.spacing(1)
        return (U, S, V.T)

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def rand(self):
        u = self._stiefel_m.rand()
        s = np.sort(np.random.rand(self._k))[::-1]
        vt = self._stiefel_n.rand().T
        return (u, s, vt)

    def _tangent(self, X, Z):
        """
        Given Z in tangent vector format, projects the components Up and Vp
        such that they satisfy the tangent space constraints up to numerical
        errors. If Z was indeed a tangent vector at X, this should barely
        affect Z (it would not at all if we had infinite numerical accuracy).
        """
        Up = Z[0] - np.dot(X[0], np.dot(X[0].T, Z[0]))
        Vp = Z[2] - np.dot(X[2].T, np.dot(X[2], Z[2]))

        return _FixedRankTangentVector((Up, Z[1], Vp))

    def randvec(self, X):
        Up = np.random.randn(self._m, self._k)
        Vp = np.random.randn(self._n, self._k)
        M = np.random.randn(self._k, self._k)

        Z = self._tangent(X, (Up, M, Vp))

        nrm = self.norm(X, Z)

        return _FixedRankTangentVector((Z[0]/nrm, Z[1]/nrm, Z[2]/nrm))

    def tangent2ambient(self, X, Z):
        """
        Transforms a tangent vector Z represented as a structure (Up, M, Vp)
        into a structure with fields (U, S, V) that represents that same
        tangent vector in the ambient space of mxn matrices, as U*S*V'.
        This matrix is equal to X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'. The
        latter is an mxn matrix, which could be too large to build
        explicitly, and this is why we return a low-rank representation
        instead. Note that there are no guarantees on U, S and V other than
        that USV' is the desired matrix. In particular, U and V are not (in
        general) orthonormal and S is not (in general) diagonal.
        (In this implementation, S is identity, but this might change.)
        """
        U = np.hstack((np.dot(X[0], Z[1]) + Z[0], X[0]))
        S = np.eye(2 * self._k)
        V = np.hstack(([X[2].T, Z[2]]))
        return (U, S, V)

    # Comment from Manopt:
    # New vector transport on June 24, 2014 (as indicated by Bart)
    # Reference: Absil, Mahony, Sepulchre 2008 section 8.1.3:
    # For Riemannian submanifolds of a Euclidean space, it is acceptable to
    # transport simply by orthogonal projection of the tangent vector
    # translated in the ambient space.
    def transp(self, X1, X2, G):
        return self.proj(X2, self.tangent2ambient(X1, G))

    def exp(self, X, U):
        raise NotImplementedError

    def log(self, X, Y):
        raise NotImplementedError

    def pairmean(self, X, Y):
        raise NotImplementedError

    def zerovec(self, X):
        return _FixedRankTangentVector((np.zeros((self._m, self._k)),
                                        np.zeros((self._k, self._k)),
                                        np.zeros((self._n, self._k))))


class _FixedRankTangentVector(tuple, ndarraySequenceMixin):
    def __repr__(self):
        repr_ = super(_FixedRankTangentVector, self).__repr__()
        return "_FixedRankTangentVector: " + repr_

    def to_ambient(self, x):
        Z1 = x[0].dot(self[1].dot(x[2]))
        Z2 = self[0].dot(x[2])
        Z3 = x[0].dot(self[2].T)
        return Z1 + Z2 + Z3

    def __add__(self, other):
        return _FixedRankTangentVector((s + o for (s, o) in zip(self, other)))

    def __sub__(self, other):
        return _FixedRankTangentVector((s - o for (s, o) in zip(self, other)))

    def __mul__(self, other):
        return _FixedRankTangentVector((other * s for s in self))

    __rmul__ = __mul__

    def __div__(self, other):
        return _FixedRankTangentVector((val / other for val in self))

    def __neg__(self):
        return _FixedRankTangentVector((-val for val in self))
