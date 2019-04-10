import abc

import numpy as np


class Manifold(object):
    """
    Abstract base class setting out a template for manifold classes. If you
    would like to extend Pymanopt with a new manifold, then your manifold
    should inherit from this class.

    Not all methods are required by all solvers. In particular, first order
    gradient based solvers such as
    :py:mod:`pymanopt.solvers.steepest_descent` and
    :py:mod:`pymanopt.solvers.conjugate_gradient` require
    :py:func:`egrad2rgrad` to be implemented but not :py:func:`ehess2rhess`.
    Second order solvers such as :py:mod:`pymanopt.solvers.trust_regions`
    will require :py:func:`ehess2rhess`.

    All of these methods correspond closely to methods in
    `Manopt <http://www.manopt.org>`_. See
    http://www.manopt.org/tutorial.html#manifolds for more details on manifolds
    in Manopt, which are effectively identical to those in Pymanopt (all of the
    methods in this class have equivalents in Manopt with the same name).
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __str__(self):
        """
        Name of the manifold
        """

    @abc.abstractproperty
    def dim(self):
        """
        Dimension of the manifold
        """

    @abc.abstractproperty
    def typicaldist(self):
        """
        Returns the "scale" of the manifold. This is used by the
        trust-regions solver, to determine default initial and maximal
        trust-region radii.
        """

    @abc.abstractmethod
    def dist(self, X, Y):
        """
        Geodesic distance on the manifold
        """

    @abc.abstractmethod
    def inner(self, X, G, H):
        """
        Inner product (Riemannian metric) on the tangent space
        """

    @abc.abstractmethod
    def proj(self, X, G):
        """
        Project into the tangent space. Usually the same as egrad2rgrad
        """

    def egrad2rgrad(self, X, G):
        """
        A mapping from the Euclidean gradient G into the tangent space
        to the manifold at X. For embedded manifolds, this is simply the
        projection of G on the tangent space at X.
        """
        return self.proj(X, G)

    @abc.abstractmethod
    def ehess2rhess(self, X, Hess):
        """
        Convert Euclidean into Riemannian Hessian.
        """

    @abc.abstractmethod
    def retr(self, X, G):
        """
        A retraction mapping from the tangent space at X to the manifold.
        See Absil for definition of retraction.
        """

    @abc.abstractmethod
    def norm(self, X, G):
        """
        Compute the norm of a tangent vector G, which is tangent to the
        manifold at X.
        """

    @abc.abstractmethod
    def rand(self):
        """
        A function which returns a random point on the manifold.
        """

    @abc.abstractmethod
    def randvec(self, X):
        """
        Returns a random, unit norm vector in the tangent space at X.
        """

    @abc.abstractmethod
    def transp(self, x1, x2, d):
        """
        Transports d, which is a tangent vector at x1, into the tangent
        space at x2.
        """

    @abc.abstractmethod
    def exp(self, X, U):
        """
        The exponential (in the sense of Lie group theory) of a tangent
        vector U at X.
        """

    @abc.abstractmethod
    def log(self, X, Y):
        """
        The logarithm (in the sense of Lie group theory) of Y. This is the
        inverse of exp.
        """

    @abc.abstractmethod
    def pairmean(self, X, Y):
        """
        Computes the intrinsic mean of X and Y, that is, a point that lies
        mid-way between X and Y on the geodesic arc joining them.
        """

    def zerovec(self, X):
        """
        Returns the zero tangent vector at X.
        """
        return np.zeros(np.shape(X))
