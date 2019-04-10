"""
Module containing manifolds of n-dimensional rotations
"""

from __future__ import division

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from scipy.misc import comb
from scipy.linalg import expm, logm

from pymanopt.tools.multi import multiprod, multitransp, multisym, multiskew
from pymanopt.manifolds.manifold import Manifold


class Rotations(Manifold):
    """
    Returns a manifold structure to optimize over rotation matrices.

    manifold = Rotations(n)
    manifold = Rotations(n, k)

    Special orthogonal group (the manifold of rotations): deals with matrices
    X of size k x n x n (or n x n if k = 1, which is the default) such that
    each n x n matrix is orthogonal, with determinant 1, i.e.,
    dot(X.T, X) = eye(n) if k = 1, or dot(X[i].T, X[i]) = eye(n) if k > 1.

    This is a description of SO(n)^k with the induced metric from the
    embedding space (R^nxn)^k, i.e., this manifold is a Riemannian
    submanifold of (R^nxn)^k endowed with the usual trace inner product.

    Tangent vectors are represented in the Lie algebra, i.e., as skew
    symmetric matrices. Use the function manifold.tangent2ambient(X, H) to
    switch from the Lie algebra representation to the embedding space
    representation. This is often necessary when defining
    problem.ehess(X, H).

    By default, the retraction is only a first-order approximation of the
    exponential. To force the use of a second-order approximation, call
    manifold.retr = manifold.retr2 after creating M. This switches from a
    QR-based computation to an SVD-based computation.

    By default, k = 1.

    Example. Based on the example found at:
    http://www.manopt.org/manifold_documentation_rotations.html

    >>> import numpy as np
    >>> from pymanopt import Problem
    >>> from pymanopt.solvers import TrustRegions
    >>> from pymanopt.manifolds import Rotations

    Generate the problem data.
    >>> n = 3
    >>> m = 10
    >>> A = np.random.randn(n, m)
    >>> B = np.random.randn(n, m)
    >>> ABt = np.dot(A,B.T)

    Create manifold - SO(n).
    >>> manifold = Rotations(n)

    Define the cost function.
    >>> cost = lambda X : -np.tensordot(X, ABt, axes=X.ndim)

    Define and solve the problem.
    >>> problem = Problem(manifold=manifold, cost=cost)
    >>> solver = TrustRegions()
    >>> X = solver.solve(problem)

    See also: Stiefel

    This file is based on rotationsfactory from Manopt: www.manopt.org
    Ported by: Lars Tingelstad
    Original author: Nicolas Boumal, Dec. 30, 2012.
    """

    def __init__(self, n, k=1):
        if k == 1:
            self._name = 'Rotations manifold SO({n})'.format(n=n)
        elif k > 1:
            self._name = 'Rotations manifold SO({n})^{k}'.format(n=n, k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._n = n
        self._k = k

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._k * comb(self._n, 2)

    def inner(self, X, U, V):
        return np.tensordot(U, V, axes=U.ndim)

    def norm(self, X, U):
        return la.norm(U)

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def proj(self, X, H):
        return multiskew(multiprod(multitransp(X), H))

    def tangent(self, X, H):
        return multiskew(H)

    def tangent2ambient(self, X, U):
        return multiprod(X, U)

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        Xt = multitransp(X)
        Xtegrad = multiprod(Xt, egrad)
        symXtegrad = multisym(Xtegrad)
        Xtehess = multiprod(Xt, ehess)
        return multiskew(Xtehess - multiprod(H, symXtegrad))

    def retr(self, X, U):
        def retri(Y):
            Q, R = la.qr(Y)
            return np.dot(Q, np.diag(np.sign(np.sign(np.diag(R)) + 0.5)))

        Y = X + multiprod(X, U)
        if self._k == 1:
            return retri(Y)
        else:
            for i in range(self._k):
                Y[i] = retri(Y[i])
            return Y

    def retr2(self, X, U):
        def retr2i(Y):
            U, _, Vt = la.svd(Y)
            return np.dot(U, Vt)

        Y = X + multiprod(X, U)
        if self._k == 1:
            return retr2i(Y)
        else:
            for i in range(self._k):
                Y[i] = retr2i(Y[i])
        return Y

    def exp(self, X, U):
        expU = U
        if self._k == 1:
            return multiprod(X, expm(expU))
        else:
            for i in range(self._k):
                expU[i] = expm(expU[i])
            return multiprod(X, expU)

    def log(self, X, Y):
        U = multiprod(multitransp(X), Y)
        if self._k == 1:
            return multiskew(np.real(logm(U)))
        else:
            for i in range(self._k):
                U[i] = np.real(logm(U[i]))
        return multiskew(U)

    def rand(self):
        return randrot(self._n, self._k)

    def randvec(self, X):
        U = randskew(self._n, self._k)
        nrmU = np.sqrt(np.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._n))
        else:
            return np.zeros((self._k, self._n, self._n))

    def transp(self, x1, x2, d):
        return d

    def pairmean(self, X, Y):
        V = self.log(X, Y)
        Y = self.exp(X, 0.5 * V)
        return Y

    def dist(self, x, y):
        return self.norm(x, self.log(x, y))


def randrot(n, N=1):

    if n == 1:
        return np.ones((N, 1, 1))

    R = np.zeros((N, n, n))

    for i in range(N):
        # Generated as such, Q is uniformly distributed over O(n), the set
        # of orthogonal matrices.
        A = rnd.randn(n, n)
        Q, RR = la.qr(A)
        Q = np.dot(Q, np.diag(np.sign(np.diag(RR))))  # Mezzadri 2007

        # If Q is in O(n) but not in SO(n), we permute the two first
        # columns of Q such that det(new Q) = -det(Q), hence the new Q will
        # be in SO(n), uniformly distributed.
        if la.det(Q) < 0:
            Q[:, [0, 1]] = Q[:, [1, 0]]

        R[i] = Q

    if N == 1:
        R = R.reshape(n, n)

    return R


def randskew(n, N=1):
    idxs = np.triu_indices(n, 1)
    S = np.zeros((N, n, n))
    for i in range(N):
        S[i][idxs] = rnd.randn(int(n * (n - 1) / 2))
        S = S - multitransp(S)
    if N == 1:
        return S.reshape(n, n)
    return S
