# References, taken from trustregions.m in manopt:

# Please cite the Manopt paper as well as the research paper:
#     @Article{genrtr,
#       Title    = {Trust-region methods on {Riemannian} manifolds},
#       Author   = {Absil, P.-A. and Baker, C. G. and Gallivan, K. A.},
#       Journal  = {Foundations of Computational Mathematics},
#       Year     = {2007},
#       Number   = {3},
#       Pages    = {303--330},
#       Volume   = {7},
#       Doi      = {10.1007/s10208-005-0179-9}
#     }
#
# See also: steepestdescent conjugategradient manopt/examples

# An explicit, general listing of this algorithm, with preconditioning,
# can be found in the following paper:
#     @Article{boumal2015lowrank,
#       Title   = {Low-rank matrix completion via preconditioned optimization
#                   on the {G}rassmann manifold},
#       Author  = {Boumal, N. and Absil, P.-A.},
#       Journal = {Linear Algebra and its Applications},
#       Year    = {2015},
#       Pages   = {200--239},
#       Volume  = {475},
#       Doi     = {10.1016/j.laa.2015.02.027},
#     }

# When the Hessian is not specified, it is approximated with
# finite-differences of the gradient. The resulting method is called
# RTR-FD. Some convergence theory for it is available in this paper:
# @incollection{boumal2015rtrfd
# 	author={Boumal, N.},
# 	title={Riemannian trust regions with finite-difference Hessian
#                      approximations are globally convergent},
# 	year={2015},
# 	booktitle={Geometric Science of Information}
# }


# This file is part of Manopt: www.manopt.org.
# This code is an adaptation to Manopt of the original GenRTR code:
# RTR - Riemannian Trust-Region
# (c) 2004-2007, P.-A. Absil, C. G. Baker, K. A. Gallivan
# Florida State University
# School of Computational Science
# (http://www.math.fsu.edu/~cbaker/GenRTR/?page=download)
# See accompanying license file.
# The adaptation was executed by Nicolas Boumal.

# Ported to pymanopt by Jamie Townsend. January 2016.

from __future__ import print_function, division

import time

import numpy as np

from pymanopt.solvers.solver import Solver

if not hasattr(__builtins__, "xrange"):
    xrange = range


class TrustRegions(Solver):
    (NEGATIVE_CURVATURE, EXCEEDED_TR, REACHED_TARGET_LINEAR,
     REACHED_TARGET_SUPERLINEAR, MAX_INNER_ITER, MODEL_INCREASED) = range(6)
    TCG_STOP_REASONS = {
        NEGATIVE_CURVATURE: "negative curvature",
        EXCEEDED_TR: "exceeded trust region",
        REACHED_TARGET_LINEAR: "reached target residual-kappa (linear)",
        REACHED_TARGET_SUPERLINEAR: "reached target residual-theta "
                                    "(superlinear)",
        MAX_INNER_ITER: "maximum inner iterations",
        MODEL_INCREASED: "model increased"
    }

    def __init__(self, miniter=3, kappa=0.1, theta=1.0, rho_prime=0.1,
                 use_rand=False, rho_regularization=1e3, *args, **kwargs):
        """
        Trust regions algorithm based on trustregions.m from the
        Manopt MATLAB package.

        Also included is the Truncated (Steihaug-Toint) Conjugate-Gradient
        algorithm, based on tCG.m from the Manopt MATLAB package.
        """
        super(TrustRegions, self).__init__(*args, **kwargs)

        self.miniter = miniter
        self.kappa = kappa
        self.theta = theta
        self.rho_prime = rho_prime
        self.use_rand = use_rand
        self.rho_regularization = rho_regularization

    def solve(self, problem, x=None, mininner=1, maxinner=None,
              Delta_bar=None, Delta0=None):
        man = problem.manifold
        verbosity = problem.verbosity

        if maxinner is None:
            maxinner = man.dim

        # Set default Delta_bar and Delta0 separately to deal with additional
        # logic: if Delta_bar is provided but not Delta0, let Delta0
        # automatically be some fraction of the provided Delta_bar.
        if Delta_bar is None:
            try:
                Delta_bar = man.typicaldist
            except NotImplementedError:
                Delta_bar = np.sqrt(man.dim)
        if Delta0 is None:
            Delta0 = Delta_bar / 8

        cost = problem.cost
        grad = problem.grad
        hess = problem.hess

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initializations
        time0 = time.time()

        # k counts the outer (TR) iterations. The semantic is that k counts the
        # number of iterations fully executed so far.
        k = 0

        # Initialize solution and companion measures: f(x), fgrad(x)
        fx = cost(x)
        fgradx = grad(x)
        norm_grad = man.norm(x, fgradx)

        # Initialize the trust region radius
        Delta = Delta0

        # To keep track of consecutive radius changes, so that we can warn the
        # user if it appears necessary.
        consecutive_TRplus = 0
        consecutive_TRminus = 0

        # ** Display:
        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            print("{:44s}f: {:+.6e}   |grad|: {:.6e}".format(
                " ", float(fx), norm_grad))

        self._start_optlog()

        while True:
            # *************************
            # ** Begin TR Subproblem **
            # *************************

            # Determine eta0
            if not self.use_rand:
                # Pick the zero vector
                eta = man.zerovec(x)
            else:
                # Random vector in T_x M (this has to be very small)
                eta = 1e-6 * man.randvec(x)
                # Must be inside trust region
                while man.norm(x, eta) > Delta:
                    eta = np.sqrt(np.sqrt(np.spacing(1)))

            # Solve TR subproblem approximately
            eta, Heta, numit, stop_inner = self._truncated_conjugate_gradient(
                problem, x, fgradx, eta, Delta, self.theta, self.kappa,
                mininner, maxinner)

            srstr = self.TCG_STOP_REASONS[stop_inner]

            # If using randomized approach, compare result with the Cauchy
            # point. Convergence proofs assume that we achieve at least (a
            # fraction of) the reduction of the Cauchy point. After this
            # if-block, either all eta-related quantities have been changed
            # consistently, or none of them have.

            if self.use_rand:
                used_cauchy = False
                # Check the curvature
                Hg = hess(x, fgradx)
                g_Hg = man.inner(x, fgradx, Hg)
                if g_Hg <= 0:
                    tau_c = 1
                else:
                    tau_c = min(norm_grad ** 3 / (Delta * g_Hg), 1)

                # and generate the Cauchy point.
                eta_c = -tau_c * Delta / norm_grad * fgradx
                Heta_c = -tau_c * Delta / norm_grad * Hg

                # Now that we have computed the Cauchy point in addition to the
                # returned eta, we might as well keep the best of them.
                mdle = (fx + man.inner(x, fgradx, eta) +
                        0.5 * man.inner(x, Heta, eta))
                mdlec = (fx + man.inner(x, fgradx, eta_c) +
                         0.5 * man.inner(x, Heta_c, eta_c))
                if mdlec < mdle:
                    eta = eta_c
                    Heta = Heta_c
                    used_cauchy = True

            # This is only computed for logging purposes, because it may be
            # useful for some user-defined stopping criteria. If this is not
            # cheap for specific applications (compared to evaluating the
            # cost), we should reconsider this.
            # norm_eta = man.norm(x, eta)

            # Compute the tentative next iterate (the proposal)
            x_prop = man.retr(x, eta)

            # Compute the function value of the proposal
            fx_prop = cost(x_prop)

            # Will we accept the proposal or not? Check the performance of the
            # quadratic model against the actual cost.
            rhonum = fx - fx_prop
            rhoden = -man.inner(x, fgradx, eta) - 0.5 * man.inner(x, eta, Heta)

            # rhonum could be anything.
            # rhoden should be nonnegative, as guaranteed by tCG, baring
            # numerical errors.

            # Heuristic -- added Dec. 2, 2013 (NB) to replace the former
            # heuristic. This heuristic is documented in the book by Conn Gould
            # and Toint on trust-region methods, section 17.4.2. rhonum
            # measures the difference between two numbers. Close to
            # convergence, these two numbers are very close to each other, so
            # that computing their difference is numerically challenging: there
            # may be a significant loss in accuracy. Since the acceptance or
            # rejection of the step is conditioned on the ratio between rhonum
            # and rhoden, large errors in rhonum result in a very large error
            # in rho, hence in erratic acceptance / rejection. Meanwhile, close
            # to convergence, steps are usually trustworthy and we should
            # transition to a Newton- like method, with rho=1 consistently. The
            # heuristic thus shifts both rhonum and rhoden by a small amount
            # such that far from convergence, the shift is irrelevant and close
            # to convergence, the ratio rho goes to 1, effectively promoting
            # acceptance of the step.  The rationale is that close to
            # convergence, both rhonum and rhoden are quadratic in the distance
            # between x and x_prop. Thus, when this distance is on the order of
            # sqrt(eps), the value of rhonum and rhoden is on the order of eps,
            # which is indistinguishable from the numerical error, resulting in
            # badly estimated rho's.
            # For abs(fx) < 1, this heuristic is invariant under offsets of f
            # but not under scaling of f. For abs(fx) > 1, the opposite holds.
            # This should not alarm us, as this heuristic only triggers at the
            # very last iterations if very fine convergence is demanded.
            rho_reg = max(1, abs(fx)) * np.spacing(1) * self.rho_regularization
            rhonum = rhonum + rho_reg
            rhoden = rhoden + rho_reg

            # This is always true if a linear, symmetric operator is used for
            # the Hessian (approximation) and if we had infinite numerical
            # precision.  In practice, nonlinear approximations of the Hessian
            # such as the built-in finite difference approximation and finite
            # numerical accuracy can cause the model to increase. In such
            # scenarios, we decide to force a rejection of the step and a
            # reduction of the trust-region radius. We test the sign of the
            # regularized rhoden since the regularization is supposed to
            # capture the accuracy to which rhoden is computed: if rhoden were
            # negative before regularization but not after, that should not be
            # (and is not) detected as a failure.
            #
            # Note (Feb. 17, 2015, NB): the most recent version of tCG already
            # includes a mechanism to ensure model decrease if the Cauchy step
            # attained a decrease (which is theoretically the case under very
            # lax assumptions). This being said, it is always possible that
            # numerical errors will prevent this, so that it is good to keep a
            # safeguard.
            #
            # The current strategy is that, if this should happen, then we
            # reject the step and reduce the trust region radius. This also
            # ensures that the actual cost values are monotonically decreasing.
            model_decreased = (rhoden >= 0)

            if not model_decreased:
                srstr = srstr + ", model did not decrease"

            try:
                rho = rhonum / rhoden
            except ZeroDivisionError:
                # Added June 30, 2015 following observation by BM.  With this
                # modification, it is guaranteed that a step rejection is
                # always accompanied by a TR reduction. This prevents
                # stagnation in this "corner case" (NaN's really aren't
                # supposed to occur, but it's nice if we can handle them
                # nonetheless).
                print("rho is NaN! Forcing a radius decrease. This should "
                      "not happen.")
                rho = np.nan

            # Choose the new TR radius based on the model performance
            trstr = "   "
            # If the actual decrease is smaller than 1/4 of the predicted
            # decrease, then reduce the TR radius.
            if rho < 1.0 / 4 or not model_decreased or np.isnan(rho):
                trstr = "TR-"
                Delta = Delta / 4
                consecutive_TRplus = 0
                consecutive_TRminus = consecutive_TRminus + 1
                if consecutive_TRminus >= 5 and verbosity >= 1:
                    consecutive_TRminus = -np.inf
                    print(" +++ Detected many consecutive TR- (radius "
                          "decreases).")
                    print(" +++ Consider decreasing options.Delta_bar "
                          "by an order of magnitude.")
                    print(" +++ Current values: Delta_bar = {:g} and "
                          "Delta0 = {:g}".format(Delta_bar, Delta0))
            # If the actual decrease is at least 3/4 of the precicted decrease
            # and the tCG (inner solve) hit the TR boundary, increase the TR
            # radius. We also keep track of the number of consecutive
            # trust-region radius increases. If there are many, this may
            # indicate the need to adapt the initial and maximum radii.
            elif rho > 3.0 / 4 and (stop_inner == self.NEGATIVE_CURVATURE or
                                    stop_inner == self.EXCEEDED_TR):
                trstr = "TR+"
                Delta = min(2 * Delta, Delta_bar)
                consecutive_TRminus = 0
                consecutive_TRplus = consecutive_TRplus + 1
                if consecutive_TRplus >= 5 and verbosity >= 1:
                    consecutive_TRplus = -np.inf
                    print(" +++ Detected many consecutive TR+ (radius "
                          "increases).")
                    print(" +++ Consider increasing options.Delta_bar "
                          "by an order of magnitude.")
                    print(" +++ Current values: Delta_bar = {:g} and "
                          "Delta0 = {:g}.".format(Delta_bar, Delta0))
            else:
                # Otherwise, keep the TR radius constant.
                consecutive_TRplus = 0
                consecutive_TRminus = 0

            # Choose to accept or reject the proposed step based on the model
            # performance. Note the strict inequality.
            if model_decreased and rho > self.rho_prime:
                # accept = True
                accstr = "acc"
                x = x_prop
                fx = fx_prop
                fgradx = grad(x)
                norm_grad = man.norm(x, fgradx)
            else:
                # accept = False
                accstr = "REJ"

            # k is the number of iterations we have accomplished.
            k = k + 1

            # ** Display:
            if verbosity == 2:
                print("{:.3s} {:.3s}   k: {:5d}     num_inner: "
                      "{:5d}     f: {:+e}   |grad|: {:e}   "
                      "{:s}".format(accstr, trstr, k, numit,
                                    float(fx), norm_grad, srstr))
            elif verbosity > 2:
                if self.use_rand and used_cauchy:
                    print("USED CAUCHY POINT")
                print("{:.3s} {:.3s}    k: {:5d}     num_inner: "
                      "{:5d}     {:s}".format(accstr, trstr, k, numit, srstr))
                print("       f(x) : {:+e}     |grad| : "
                      "{:e}".format(fx, norm_grad))
                print("        rho : {:e}".format(rho))

            # ** CHECK STOPPING criteria
            stop_reason = self._check_stopping_criterion(
                time0, gradnorm=norm_grad, iter=k)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x, fx, stop_reason, time0,
                              gradnorm=norm_grad, iter=k)
            return x, self._optlog

    def _truncated_conjugate_gradient(self, problem, x, fgradx, eta, Delta,
                                      theta, kappa, mininner, maxinner):
        man = problem.manifold
        inner = man.inner
        hess = problem.hess
        precon = problem.precon

        if not self.use_rand:  # and therefore, eta == 0
            Heta = man.zerovec(x)
            r = fgradx
            e_Pe = 0
        else:  # and therefore, no preconditioner
            # eta (presumably) ~= 0 was provided by the caller.
            Heta = hess(x, eta)
            r = fgradx + Heta
            e_Pe = inner(x, eta, eta)

        r_r = inner(x, r, r)
        norm_r = np.sqrt(r_r)
        norm_r0 = norm_r

        # Precondition the residual
        if not self.use_rand:
            z = precon(x, r)
        else:
            z = r

        # Compute z'*r
        z_r = inner(x, z, r)
        d_Pd = z_r

        # Initial search direction
        delta = -z
        if not self.use_rand:
            e_Pd = 0
        else:
            e_Pd = inner(x, eta, delta)

        # If the Hessian or a linear Hessian approximation is in use, it is
        # theoretically guaranteed that the model value decreases strictly with
        # each iteration of tCG. Hence, there is no need to monitor the model
        # value. But, when a nonlinear Hessian approximation is used (such as
        # the built-in finite-difference approximation for example), the model
        # may increase. It is then important to terminate the tCG iterations
        # and return the previous (the best-so-far) iterate. The variable below
        # will hold the model value.

        def model_fun(eta, Heta):
            return inner(x, eta, fgradx) + 0.5 * inner(x, eta, Heta)
        if not self.use_rand:
            model_value = 0
        else:
            model_value = model_fun(eta, Heta)

        # Pre-assume termination because j == end.
        stop_tCG = self.MAX_INNER_ITER

        # Begin inner/tCG loop.
        for j in xrange(0, int(maxinner)):
            # This call is the computationally intensive step
            Hdelta = hess(x, delta)

            # Compute curvature (often called kappa)
            d_Hd = inner(x, delta, Hdelta)

            # Note that if d_Hd == 0, we will exit at the next "if" anyway.
            alpha = z_r / d_Hd
            # <neweta,neweta>_P =
            # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
            e_Pe_new = e_Pe + 2 * alpha * e_Pd + alpha ** 2 * d_Pd

            # Check against negative curvature and trust-region radius
            # violation. If either condition triggers, we bail out.
            if d_Hd <= 0 or e_Pe_new >= Delta**2:
                # want
                #  ee = <eta,eta>_prec,x
                #  ed = <eta,delta>_prec,x
                #  dd = <delta,delta>_prec,x
                tau = ((-e_Pd +
                        np.sqrt(e_Pd * e_Pd +
                                d_Pd * (Delta ** 2 - e_Pe))) / d_Pd)

                eta = eta + tau * delta

                # If only a nonlinear Hessian approximation is available, this
                # is only approximately correct, but saves an additional
                # Hessian call.
                Heta = Heta + tau * Hdelta

                # Technically, we may want to verify that this new eta is
                # indeed better than the previous eta before returning it (this
                # is always the case if the Hessian approximation is linear,
                # but I am unsure whether it is the case or not for nonlinear
                # approximations.) At any rate, the impact should be limited,
                # so in the interest of code conciseness (if we can still hope
                # for that), we omit this.

                if d_Hd <= 0:
                    stop_tCG = self.NEGATIVE_CURVATURE
                else:
                    stop_tCG = self.EXCEEDED_TR
                break

            # No negative curvature and eta_prop inside TR: accept it.
            e_Pe = e_Pe_new
            new_eta = eta + alpha * delta

            # If only a nonlinear Hessian approximation is available, this is
            # only approximately correct, but saves an additional Hessian call.
            new_Heta = Heta + alpha * Hdelta

            # Verify that the model cost decreased in going from eta to
            # new_eta. If it did not (which can only occur if the Hessian
            # approximation is nonlinear or because of numerical errors), then
            # we return the previous eta (which necessarily is the best reached
            # so far, according to the model cost). Otherwise, we accept the
            # new eta and go on.
            new_model_value = model_fun(new_eta, new_Heta)
            if new_model_value >= model_value:
                stop_tCG = self.MODEL_INCREASED
                break

            eta = new_eta
            Heta = new_Heta
            model_value = new_model_value

            # Update the residual.
            r = r + alpha * Hdelta

            # Compute new norm of r.
            r_r = inner(x, r, r)
            norm_r = np.sqrt(r_r)

            # Check kappa/theta stopping criterion.
            # Note that it is somewhat arbitrary whether to check this stopping
            # criterion on the r's (the gradients) or on the z's (the
            # preconditioned gradients). [CGT2000], page 206, mentions both as
            # acceptable criteria.
            if (j >= mininner and
               norm_r <= norm_r0 * min(norm_r0**theta, kappa)):
                # Residual is small enough to quit
                if kappa < norm_r0 ** theta:
                    stop_tCG = self.REACHED_TARGET_LINEAR
                else:
                    stop_tCG = self.REACHED_TARGET_SUPERLINEAR
                break

            # Precondition the residual.
            if not self.use_rand:
                z = precon(x, r)
            else:
                z = r

            # Save the old z'*r.
            zold_rold = z_r
            # Compute new z'*r.
            z_r = inner(x, z, r)

            # Compute new search direction
            beta = z_r / zold_rold
            delta = -z + beta * delta

            # Update new P-norms and P-dots [CGT2000, eq. 7.5.6 & 7.5.7].
            e_Pd = beta * (e_Pd + alpha * d_Pd)
            d_Pd = z_r + beta * beta * d_Pd

        return eta, Heta, j, stop_tCG
