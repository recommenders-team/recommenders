from __future__ import print_function, division

import time
from copy import deepcopy

import numpy as np

from pymanopt.solvers.linesearch import LineSearchAdaptive
from pymanopt.solvers.solver import Solver
from pymanopt import tools


BetaTypes = tools.make_enum(
    "BetaTypes",
    "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split())


class ConjugateGradient(Solver):
    """
    Module containing conjugate gradient algorithm based on
    conjugategradient.m from the manopt MATLAB package.
    """

    def __init__(self, beta_type=BetaTypes.HestenesStiefel, orth_value=np.inf,
                 linesearch=None, *args, **kwargs):
        """
        Instantiate gradient solver class.
        Variable attributes (defaults in brackets):
            - beta_type (BetaTypes.HestenesStiefel)
                Conjugate gradient beta rule used to construct the new search
                direction
            - orth_value (numpy.inf)
                Parameter for Powell's restart strategy. An infinite
                value disables this strategy. See in code formula for
                the specific criterion used.
            - linesearch (LineSearchAdaptive)
                The linesearch method to used.
        """
        super(ConjugateGradient, self).__init__(*args, **kwargs)

        self._beta_type = beta_type
        self._orth_value = orth_value

        if linesearch is None:
            self._linesearch = LineSearchAdaptive()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    def solve(self, problem, x=None, reuselinesearch=False):
        """
        Perform optimization using nonlinear conjugate gradient method with
        linesearch.
        This method first computes the gradient of obj w.r.t. arg, and then
        optimizes by moving in a direction that is conjugate to all previous
        search directions.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        stepsize = np.nan
        time0 = time.time()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        # Calculate initial cost-related quantities
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        Pgrad = problem.precon(x, grad)
        gradPgrad = man.inner(x, grad, Pgrad)

        # Initial descent direction is the negative gradient
        desc_dir = -Pgrad

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'beta_type': self._beta_type,
                                         'orth_value': self._orth_value,
                                         'linesearcher': linesearch})

        while True:
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            stop_reason = self._check_stopping_criterion(
                time0, gradnorm=gradnorm, iter=iter + 1, stepsize=stepsize)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = man.inner(x, grad, desc_dir)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if verbosity >= 3:
                    print("Conjugate gradient info: got an ascent direction "
                          "(df0 = %.2f), reset to the (preconditioned) "
                          "steepest descent direction." % df0)
                # Reset to negative gradient: this discards the CG memory.
                desc_dir = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            stepsize, newx = linesearch.search(objective, man, x, desc_dir,
                                               cost, df0)

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradnorm = man.norm(newx, newgrad)
            Pnewgrad = problem.precon(newx, newgrad)
            newgradPnewgrad = man.inner(newx, newgrad, Pnewgrad)

            # Apply the CG scheme to compute the next search direction
            oldgrad = man.transp(x, newx, grad)
            orth_grads = man.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad

            # Powell's restart strategy (see page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = man.transp(x, newx, desc_dir)

                if self._beta_type == BetaTypes.FletcherReeves:
                    beta = newgradPnewgrad / gradPgrad
                elif self._beta_type == BetaTypes.PolakRibiere:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    beta = max(0, ip_diff / gradPgrad)
                elif self._beta_type == BetaTypes.HestenesStiefel:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta = max(0,
                                   ip_diff / man.inner(newx, diff, desc_dir))
                    # if ip_diff = man.inner(newx, diff, desc_dir) = 0
                    except ZeroDivisionError:
                        beta = 1
                elif self._beta_type == BetaTypes.HagerZhang:
                    diff = newgrad - oldgrad
                    Poldgrad = man.transp(x, newx, Pgrad)
                    Pdiff = Pnewgrad - Poldgrad
                    deno = man.inner(newx, diff, desc_dir)
                    numo = man.inner(newx, diff, Pnewgrad)
                    numo -= (2 * man.inner(newx, diff, Pdiff) *
                             man.inner(newx, desc_dir, newgrad) / deno)
                    beta = numo / deno
                    # Robustness (see Hager-Zhang paper mentioned above)
                    desc_dir_norm = man.norm(newx, desc_dir)
                    eta_HZ = -1 / (desc_dir_norm * min(0.01, gradnorm))
                    beta = max(beta, eta_HZ)
                else:
                    types = ", ".join(
                        ["BetaTypes.%s" % t for t in BetaTypes._fields])
                    raise ValueError(
                        "Unknown beta_type %s. Should be one of %s." % (
                            self._beta_type, types))

                desc_dir = -Pnewgrad + beta * desc_dir

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradnorm = newgradnorm
            gradPgrad = newgradPnewgrad

            iter += 1

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x, cost, stop_reason, time0,
                              stepsize=stepsize, gradnorm=gradnorm,
                              iter=iter)
            return x, self._optlog
