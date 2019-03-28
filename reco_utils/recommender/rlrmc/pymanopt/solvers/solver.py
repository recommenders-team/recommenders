import time
import abc


class Solver(object):
    '''
    Abstract base class setting out template for solver classes.
    '''

    __metaclass__ = abc.ABCMeta

    def __init__(self, maxtime=1000, maxiter=1000, mingradnorm=1e-6,
                 minstepsize=1e-10, maxcostevals=5000, logverbosity=0):
        """
        Variable attributes (defaults in brackets):
            - maxtime (1000)
                Max time (in seconds) to run.
            - maxiter (1000)
                Max number of iterations to run.
            - mingradnorm (1e-6)
                Terminate if the norm of the gradient is below this.
            - minstepsize (1e-10)
                Terminate if linesearch returns a vector whose norm is below
                this.
            - maxcostevals (5000)
                Maximum number of allowed cost evaluations
            - logverbosity (0)
                Level of information logged by the solver while it operates,
                0 is silent, 2 ist most information.
        """
        self._maxtime = maxtime
        self._maxiter = maxiter
        self._mingradnorm = mingradnorm
        self._minstepsize = minstepsize
        self._maxcostevals = maxcostevals
        self._logverbosity = logverbosity
        self._optlog = None

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def solve(self, problem, x=None):
        '''
        Solve the given :py:class:`pymanopt.core.problem.Problem` (starting
        from a random initial guess if the optional argument x is not
        provided).
        '''
        pass

    def _check_stopping_criterion(self, time0, iter=-1, gradnorm=float('inf'),
                                  stepsize=float('inf'), costevals=-1):
        reason = None
        if time.time() >= time0 + self._maxtime:
            reason = ("Terminated - max time reached after %d iterations."
                      % iter)
        elif iter >= self._maxiter:
            reason = ("Terminated - max iterations reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif gradnorm < self._mingradnorm:
            reason = ("Terminated - min grad norm reached after %d "
                      "iterations, %.2f seconds." % (
                          iter, (time.time() - time0)))
        elif stepsize < self._minstepsize:
            reason = ("Terminated - min stepsize reached after %d iterations, "
                      "%.2f seconds." % (iter, (time.time() - time0)))
        elif costevals >= self._maxcostevals:
            reason = ("Terminated - max cost evals reached after "
                      "%.2f seconds." % (time.time() - time0))
        return reason

    def _start_optlog(self, solverparams=None, extraiterfields=None):
        if self._logverbosity <= 0:
            self._optlog = None
        else:
            self._optlog = {'solver': str(self),
                            'stoppingcriteria': {'maxtime':
                                                 self._maxtime,
                                                 'maxiter':
                                                 self._maxiter,
                                                 'mingradnorm':
                                                 self._mingradnorm,
                                                 'minstepsize':
                                                 self._minstepsize,
                                                 'maxcostevals':
                                                 self._maxcostevals},
                            'solverparams': solverparams
                            }
        if self._logverbosity >= 2:
            if extraiterfields:
                self._optlog['iterations'] = {'iteration': [],
                                              'time': [],
                                              'x': [],
                                              'f(x)': []}
                for field in extraiterfields:
                    self._optlog['iterations'][field] = []

    def _append_optlog(self, iteration, x, fx, **kwargs):
        # In case not every iteration is being logged
        self._optlog['iterations']['iteration'].append(iteration)
        self._optlog['iterations']['time'].append(time.time())
        self._optlog['iterations']['x'].append(x)
        self._optlog['iterations']['f(x)'].append(fx)
        for key in kwargs:
            self._optlog['iterations'][key].append(kwargs[key])

    def _stop_optlog(self, x, objective, stop_reason, time0,
                     stepsize=float('inf'), gradnorm=float('inf'),
                     iter=-1, costevals=-1):
        self._optlog['stoppingreason'] = stop_reason
        self._optlog['final_values'] = {'x': x,
                                        'f(x)': objective,
                                        'time': time.time() - time0}
        if stepsize is not float('inf'):
            self._optlog['final_values']['stepsize'] = stepsize
        if gradnorm is not float('inf'):
            self._optlog['final_values']['gradnorm'] = gradnorm
        if iter is not -1:
            self._optlog['final_values']['iterations'] = iter
        if costevals is not -1:
            self._optlog['final_values']['costevals'] = costevals
