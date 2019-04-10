"""
Module containing functions to differentiate functions using tensorflow.
"""
try:
    import tensorflow as tf
    try:
        from tensorflow.python.ops.gradients import _hessian_vector_product
    except ImportError:
        from tensorflow.python.ops.gradients_impl import \
            _hessian_vector_product
except ImportError:
    tf = None

from ._backend import Backend, assert_backend_available


class TensorflowBackend(Backend):
    def __init__(self):
        if tf is not None:
            self._session = tf.Session()

    def __str__(self):
        return "tensorflow"

    def is_available(self):
        return tf is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        if isinstance(objective, tf.Tensor):
            if (argument is None or not
                isinstance(argument, tf.Variable) and not
                all([isinstance(arg, tf.Variable)
                     for arg in argument])):
                raise ValueError(
                    "Tensorflow backend requires an argument (or sequence of "
                    "arguments) with respect to which compilation is to be "
                    "carried out")
            return True
        return False

    @assert_backend_available
    def compile_function(self, objective, argument):
        if not isinstance(argument, list):

            def func(x):
                feed_dict = {argument: x}
                return self._session.run(objective, feed_dict)
        else:

            def func(x):
                feed_dict = {i: d for i, d in zip(argument, x)}
                return self._session.run(objective, feed_dict)

        return func

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' and return as a function.
        """
        tfgrad = tf.gradients(objective, argument)

        if not isinstance(argument, list):

            def grad(x):
                feed_dict = {argument: x}
                return self._session.run(tfgrad[0], feed_dict)

        else:

            def grad(x):
                feed_dict = {i: d for i, d in zip(argument, x)}
                return self._session.run(tfgrad, feed_dict)

        return grad

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        if not isinstance(argument, list):
            argA = tf.zeros_like(argument)
            tfhess = _hessian_vector_product(objective, [argument], [argA])

            def hess(x, a):
                feed_dict = {argument: x, argA: a}
                return self._session.run(tfhess[0], feed_dict)

        else:
            argA = [tf.zeros_like(arg) for arg in argument]
            tfhess = _hessian_vector_product(objective, argument, argA)

            def hess(x, a):
                feed_dict = {i: d for i, d in zip(argument+argA, x+a)}
                return self._session.run(tfhess, feed_dict)

        return hess
