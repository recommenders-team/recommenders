"""
Module containing functions to compile and differentiate Theano graphs. Part of
the pymanopt package.

Jamie Townsend December 2014
"""
try:
    import theano
    import theano.tensor as T
    from theano.gradient import disconnected_grad
except ImportError:
    theano = None
    T = None

from ._backend import Backend, assert_backend_available


class TheanoBackend(Backend):
    def __str__(self):
        return "theano"

    def is_available(self):
        return theano is not None and T is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        if isinstance(objective, T.TensorVariable):
            if (argument is None or not
                isinstance(argument, T.TensorVariable) and not
                all([isinstance(arg, T.TensorVariable)
                     for arg in argument])):
                raise ValueError(
                    "Theano backend requires an argument (or sequence of "
                    "arguments) with respect to which compilation is to be "
                    "carried out")
            return True
        return False

    @assert_backend_available
    def compile_function(self, objective, argument):
        """
        Wrapper for the theano.function(). Compiles a theano graph into a
        python function.
        """
        try:
            return theano.function([argument], objective)
        except TypeError:
            # Assume we are on a product manifold
            compiled = theano.function([arg for arg in argument], objective)
            return lambda x: compiled(*x)

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Wrapper for theano.tensor.grad(). Computes the gradient of 'objective'
        with respect to 'argument' and returns compiled version.
        """
        g = T.grad(objective, argument)
        return self.compile_function(g, argument)

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        """
        Computes the directional derivative of the gradient (which is equal to
        the Hessian multiplied by direction).
        """
        g = T.grad(objective, argument)

        # Create a new tensor A, which has the same type (i.e. same
        # dimensionality) as argument.
        is_product_manifold = isinstance(argument, (list, tuple))
        if not is_product_manifold:
            A = argument.type()
        else:
            A = [arg.type() for arg in argument]

        # First attempt efficient 'R-op', this directly calculates the
        # directional derivative of the gradient.
        try:
            R = T.Rop(g, argument, A)
        except NotImplementedError:
            # Implementation based on
            # tensorflow.python.ops.gradients_impl._hessian_vector_product
            if not is_product_manifold:
                proj = T.sum(g * disconnected_grad(A))
                R = T.grad(proj, argument)
            else:
                proj = [T.sum(g_elem * disconnected_grad(a_elem))
                        for g_elem, a_elem in zip(g, A)]
                proj_grad = [T.grad(proj_elem, argument,
                                    disconnected_inputs="ignore",
                                    return_disconnected="None")
                             for proj_elem in proj]
                proj_grad_transpose = map(list, zip(*proj_grad))
                proj_grad_stack = [
                    T.stacklists([c for c in row if c is not None])
                    for row in proj_grad_transpose]
                R = [T.sum(stack, axis=0) for stack in proj_grad_stack]

        if not is_product_manifold:
            hess = theano.function([argument, A], R, on_unused_input="warn")
        else:
            hess_prod = theano.function(argument + A, R,
                                        on_unused_input="warn")

            def hess(x, a):
                return hess_prod(*(x + a))

        return hess
