from __future__ import print_function, division

import time

import numpy as np

from pymanopt import Problem
from pymanopt.solvers.steepest_descent import SteepestDescent
from pymanopt.solvers.solver import Solver


def compute_centroid(man, x):
    """
    Compute the centroid as Karcher mean of points x belonging to the manifold
    man.
    """
    n = len(x)

    def objective(y):  # weighted Frechet variance
        acc = 0
        for i in range(n):
            acc += man.dist(y, x[i]) ** 2
        return acc / 2

    def gradient(y):
        g = man.zerovec(y)
        for i in range(n):
            g -= man.log(y, x[i])
        return g

    # TODO: manopt runs a few TR iterations here. For us to do this, we either
    #       need to work out the Hessian of the Frechet variance by hand or
    #       implement approximations for the Hessian to use in the TR solver.
    #       This is because we cannot implement the Frechet variance with
    #       theano and compute the Hessian automatically due to dependency on
    #       the manifold-dependent distance function.
    solver = SteepestDescent(maxiter=15)
    problem = Problem(man, cost=objective, grad=gradient, verbosity=0)
    return solver.solve(problem)


class NelderMead(Solver):
    """
    Nelder-Mead minimization alglorithm for derivative-free minimization
    based on neldermead.m and centroid.m from the manopt MATLAB package.
    """

    def __init__(self, maxcostevals=None, maxiter=None, reflection=1,
                 expansion=2, contraction=0.5, *args, **kwargs):
        """
        Instantiate Nelder-Mead method solver class.
        Variable attributes (defaults in brackets):
            - maxcostevals (max(5000, 2 * dim))
                Maximum number of allowed cost evaluations
            - maxiter (max(500, 4 * dim))
                Maximum number of allowed iterations
            - reflection (1)
                Determines how far to reflect away from the worst vertex;
                stretched (reflection > 1), compressed (0 < reflection < 1),
                or exact (reflection = 1)
            - expansion (2)
                Factor by which to expand the reflected simplex
            - contraction (0.5)
                Factor by which to contract the reflected simplex
        """
        super(NelderMead, self).__init__(*args, **kwargs)

        self._maxcostevals = maxcostevals
        self._maxiter = maxiter
        self._reflection = reflection
        self._expansion = expansion
        self._contraction = contraction

    def solve(self, problem, x=None):
        """
        Perform optimization using a Nelder-Mead minimization algorithm.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Initial population of elements on the
                manifold. If None then an initial population will be randomly
                generated
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = man.dim
        if self._maxcostevals is None:
            self._maxcostevals = max(1000, 2 * dim)
        if self._maxiter is None:
            self._maxiter = max(2000, 4 * dim)

        # If no initial simplex x is given by the user, generate one at random.
        if x is None:
            x = [man.rand() for i in range(int(dim + 1))]
        elif not hasattr(x, "__iter__"):
            raise ValueError("The initial simplex x must be iterable")
        else:
            # XXX: Is this necessary?
            if len(x) != dim + 1:
                print("The simplex size was adapted to the dimension "
                      "of the manifold")
                x = x[:dim + 1]

        # Compute objective-related quantities for x, and setup a function
        # evaluations counter.
        costs = np.array([objective(xi) for xi in x])
        costevals = dim + 1

        # Sort simplex points by cost.
        order = np.argsort(costs)
        costs = costs[order]
        x = [x[i] for i in order]  # XXX: Probably inefficient

        # Iteration counter (at any point, iter is the number of fully executed
        # iterations so far).
        iter = 0

        time0 = time.time()

        self._start_optlog()

        while True:
            iter += 1

            if verbosity >= 2:
                print("Cost evals: %7d\t"
                      "Best cost: %+.8e" % (costevals, costs[0]))

            # Sort simplex points by cost.
            order = np.argsort(costs)
            costs = costs[order]
            x = [x[i] for i in order]  # XXX: Probably inefficient

            stop_reason = self._check_stopping_criterion(
                time0, iter=iter, costevals=costevals)
            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

            # Compute a centroid for the dim best points.
            xbar = compute_centroid(man, x[:-1])

            # Compute the direction for moving along the axis xbar - worst x.
            vec = man.log(xbar, x[-1])

            # Reflection step
            xr = man.exp(xbar, -self._reflection * vec)
            costr = objective(xr)
            costevals += 1

            # If the reflected point is honorable, drop the worst point,
            # replace it by the reflected point and start a new iteration.
            if costr >= costs[0] and costr < costs[-2]:
                if verbosity >= 2:
                    print("Reflection")
                costs[-1] = costr
                x[-1] = xr
                continue

            # If the reflected point is better than the best point, expand.
            if costr < costs[0]:
                xe = man.exp(xbar, -self._expansion * vec)
                coste = objective(xe)
                costevals += 1
                if coste < costr:
                    if verbosity >= 2:
                        print("Expansion")
                    costs[-1] = coste
                    x[-1] = xe
                    continue
                else:
                    if verbosity >= 2:
                        print("Reflection (failed expansion)")
                    costs[-1] = costr
                    x[-1] = xr
                    continue

            # If the reflected point is worse than the second to worst point,
            # contract.
            if costr >= costs[-2]:
                if costr < costs[-1]:
                    # do an outside contraction
                    xoc = man.exp(xbar, -self._contraction * vec)
                    costoc = objective(xoc)
                    costevals += 1
                    if costoc <= costr:
                        if verbosity >= 2:
                            print("Outside contraction")
                        costs[-1] = costoc
                        x[-1] = xoc
                        continue
                else:
                    # do an inside contraction
                    xic = man.exp(xbar, self._contraction * vec)
                    costic = objective(xic)
                    costevals += 1
                    if costic <= costs[-1]:
                        if verbosity >= 2:
                            print("Inside contraction")
                        costs[-1] = costic
                        x[-1] = xic
                        continue

            # If we get here, shrink the simplex around x[0].
            if verbosity >= 2:
                print("Shrinkage")
            x0 = x[0]
            for i in np.arange(1, dim + 1):
                x[i] = man.pairmean(x0, x[i])
                costs[i] = objective(x[i])
            costevals += dim

        if self._logverbosity <= 0:
            return x[0]
        else:
            self._stop_optlog(x[0], objective(x[0]), stop_reason, time0,
                              costevals=costevals, iter=iter)
            return x[0], self._optlog
