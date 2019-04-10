"""
Module containing tools for testing correctness in Pymanopt. Note, these
currently require autograd.

Note: the methods for generating rgrad, egrad2rgrad, ehess and ehess2rhess
will only be correct if the manifold is a submanifold of Euclidean space,
that is if the projection is an orthogonal projection onto the tangent space.
"""
import numpy as np

from autograd import grad, jacobian


def rgrad(cost, proj):
    """
    Generates the Riemannain gradient of cost. Cost must be defined using
    autograd.numpy.
    """
    return lambda x: proj(x, grad(cost)(x))


def egrad2rgrad(proj):
    """
    Generates an egrad2rgrad function.
    """
    return lambda x, g: proj(x, g)


def rhess(cost, proj):
    """
    Generates the Riemannian hessian of the cost. Specifically, rhess(cost,
    proj)(x, u) is the directional derivatative of cost at point X on the
    manifold, in direction u.
    cost and proj must be defined using autograd.numpy.
    See http://sites.uclouvain.be/absil/2013-01/Weingarten_07PA_techrep.pdf
    for some discussion.
    proj and cost must be defined using autograd.
    Currently this is correct but not efficient, because of the jacobian-
    vector product. Hopefully this can be fixed in future.
    """
    return lambda x, u: proj(x, np.tensordot(jacobian(rgrad(cost, proj))(x), u,
                                             axes=u.ndim))


def ehess2rhess(proj):
    """
    Generates an ehess2rhess function for a manifold which is a sub-manifold
    of Euclidean space.
    ehess2rhess(proj)(x, egrad, ehess, u) converts the Euclidean hessian ehess
    at the point x to a Riemannian hessian. That is the directional
    derivatative of the gradient in the direction u.
    proj must be defined using autograd.numpy.
    This will not be an efficient implementation because of missing support
    for efficient jacobian-vector products in autograd.
    """
    # Differentiate proj w.r.t. the first argument
    d_proj = jacobian(proj)
    return lambda x, egrad, ehess, u: proj(x, ehess +
                                           np.tensordot(d_proj(x, egrad), u,
                                                        axes=u.ndim))
