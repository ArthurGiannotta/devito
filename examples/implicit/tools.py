from devito.symbolics import retrieve_functions
from devito.tools import as_tuple
from devito.types import Eq
from sympy import poly, expand

__all__ = ['collect', 'collect_shifted', 'get_shifted']


def collect(eq, targets):

    """
    Algebraically collect terms of an Eq w.r.t. given symbols.

    Parameters
    ----------
    eq : expr-like
        The equation to have its terms collected.
    targets : list of symbols
        The symbols to be collected at the left hand side. May be an array of `Function`
        or any other symbolic object.
    """

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    # Make a polynomial of the terms to be collected then find its coefficients
    coeffs = poly(expand(eq.evaluate), targets).coeffs()

    rhs = -coeffs[-1]
    lhs = 0
    for i in range(len(targets)):
        lhs += coeffs[i] * targets[i]

    return Eq(lhs, rhs)


def get_shifted(eq, shift=1, funcs=None):
    if funcs is None:
        funcs = getattr(eq.lhs, "_functions", frozenset())
        funcs = funcs.union(getattr(eq.rhs, "_functions", frozenset()))

    funcs = as_tuple(funcs)
    functions = retrieve_functions(eq.evaluate)
    shift = as_tuple(shift)

    assert len(funcs) > 0, "Can't collect shifted terms of an equation with no functions"

    # Warning: It is assumed that all functions have the same time dimension
    time_dimension = funcs[0].dimensions[funcs[0]._time_position]
    shifted_dimensions = [time_dimension + s * time_dimension.spacing for s in shift]

    funcs = as_tuple([f.func for f in funcs])
    shifted_terms = set()
    for function in functions:
        if function.func in funcs and \
           function.args[function._time_position] in shifted_dimensions:
            shifted_terms.add(function)
    shifted_terms = as_tuple(shifted_terms)

    return shifted_terms


def collect_shifted(eq, shift=1, funcs=None):

    """
    Algebraically collect terms of an Eq that are shifted in time like, for example,
    u(t + 2Δt, x, y) or v(t - Δt, x - Δx, y). Also returns the

    Parameters
    ----------
    eq : expr-like
        The equation to have its terms shifted in time collected.
    shift : int or tuple of int, optional
        The shift numbers. If shift = (-3, 2), for example, all functions of the form
        u(t - 3Δt, x, y) or u(t + 2Δt, x, y) will be collected on the lhs. Defaults to 1.
    func : Function or tuple of Function, optional
        The functions that will be collected. Defaults to all functions found in the Eq.
    """

    return collect(eq, get_shifted(eq, shift, funcs))
