from .tools import collect, get_shifted
from devito.types import (ConditionalDimension, DefaultDimension, Dimension, Eq,
                          Function, Inc, solve)
from enum import Enum, auto
from math import pi
from numpy import int32
from sympy import oo, Abs

__all__ = ['IterativeMethod']


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


class IterativeMethod(Enum):
    JACOBI = auto()
    GAUSS = auto()
    GAUSS_SEIDEL = GAUSS
    SOR = auto()
    SUCCESSIVE_OVER_RELAXATION = SOR

    @static_vars(constants=dict())
    def set_constant(self, name, value):

        """
        Set a constant related to an iterative method.

        Constants
        ---------
        p : positive real
            The p-norm constant. The residual is calculated as |RHS - LHS| ** p.
            Defaults to 1.
        epsilon : positive real
            The upper limit for the residual. The iteration loop breaks whenever
            residual < epsilon. Defaults to ∞ (force only one iteration).
        omega : expr-like
            The ω constant for the SOR method. Defaults to 2 - 2 π hx.

        Examples
        --------
        >>> IterativeMethod.JACOBI.set_constant('p', 2)
        >>> IterativeMethod.GAUSS.set_constant('epsilon', 0.1)

        >>> from sympy import sin
        >>> hx = grid.spacing_symbols[0]
        >>> IterativeMethod.SOR.set_constant('omega', 2 / (1 + pi * sin(hx)))
        """

        IterativeMethod.set_constant.constants[name] = value

    @static_vars(tempi=0)
    def explicit_expressions(self, equation, function=None, shift=1):

        """
        Get the explicit expressions equivalent to the application of an implicit method.
        These equations can be passed to an Operator that will apply the chosen method.

        The supported methods for this class are JACOBI, GAUSS and SOR.

        A = [a11 a12 ... a1n   B = [b1
             a21 a22 ... a2n        b2
             .    .  .    .          .
             .    .   .   .          .
             .    .    .  .          .
             an1 an2 ... ann]       bn]

        A x = B

        JACOBI:       x(k+1) = (bi - Σ(aij xj(k))) / aii
        GAUSS: x(k+1) = (bi - Σ(aij xj(k+1))) / aii
        SOR:          x(k+1) = ω * (bi - Σ(aij xj(k+1))) / aii + (1 - ω) x(k)

        Parameters
        ----------
        equation : Eq
            The implicit equation to be converted into explicit ones.
        shift : int
            The time shift to solve for. If shift = 2, for example, then
            x(t + 2Δt) = f(x(t + Δt), x(t), x(t - Δt), ...). For equations evolving
            backwards in time, the shift is negative. Defaults to 1.

        Examples
        --------
        >>> # u is a TimeFunction, c is a real scalar
        >>> pde = Eq(u.dtr + c * (u.forward.dxl + u.forward.dyl)) # 2D Wave Equation
        >>> exprs = IterativeMethod.SOR.explicit_expressions(pde)
        """

        if function is None:
            function = next(iter(equation.lhs._functions))
        shifted = get_shifted(equation, shift)
        equation = collect(equation, shifted)

        time_dim = function.dimensions[function._time_position]
        shifted = function.subs(time_dim, time_dim + shift * time_dim.spacing)

        # The variable that will store which iteration is currently being executed
        niter = Function(name='niter', shape=(1,),
                         dimensions=(Dimension(name='niter_dim'),), dtype=int32)

        constants = IterativeMethod.set_constant.constants
        tempi = str(IterativeMethod.explicit_expressions.tempi)
        IterativeMethod.explicit_expressions.tempi += 1

        # Dimension in which to execute the iterations
        iter_dim = Dimension(name='iter')

        # Create a new set of "temporary" spatial dimensions
        space_dims = list(function.dimensions)
        space_shapes = list(function.shape)
        del space_dims[function._time_position]
        del space_shapes[function._time_position]

        temp_dims, orig_to_temp_subs, temp_to_orig_subs = [], [], []
        for dimension, shape in zip(space_dims, space_shapes):
            temp_dimension = DefaultDimension('tempdim_' + dimension.name + tempi,
                                              default_value=shape)
            temp_dims += [temp_dimension]

            orig_to_temp_subs += [(dimension, temp_dimension)]
            temp_to_orig_subs += [(temp_dimension, dimension)]

        if self.value == self.JACOBI.value or \
           self.value == self.GAUSS.value or \
           self.value == self.SOR.value:
            residual = Function(name='residual' + tempi, shape=(1,),
                                dimensions=(Dimension(name='residual_dim',),))
            temp = Function(name='temp' + tempi, shape=function.grid.shape,
                            dimensions=space_dims)

            # This expression calculates the b coefficients
            tempEq = Eq(temp, equation.rhs)

            # This is the expression for the right hand side of the Gauss-Seidel method,
            # which will be used as a basis to get Jacobi and SOR expressions
            gaussEq = solve(equation.lhs - temp, shifted)

            epsilon = constants.get('epsilon', oo)
            p = constants.get('p', 1)

            if self.value == self.JACOBI.value:
                # TODO: Implement Jacobi
                # This is actually kind of hard since a temporary buffer must be created
                raise NotImplementedError
            elif self.value == self.GAUSS.value:
                # This is the Gauss Seidel expression
                iterEq = Eq(shifted, gaussEq, implicit_dims=[iter_dim] + temp_dims)
            elif self.value == self.SOR.value:
                hx = function.grid.spacing_symbols[0]
                omega = constants.get('omega', 2 - 2 * pi * hx)
                sorEq = omega * gaussEq + (1 - omega) * function.forward

                # This is the SOR expression
                iterEq = Eq(shifted, sorEq, implicit_dims=[iter_dim] + space_dims)

            # This is the expression for the residual
            res = (iterEq.rhs - iterEq.lhs).subs(orig_to_temp_subs)
            resEq = [Eq(residual[0], 0, implicit_dims=[time_dim, iter_dim]),
                     Inc(residual[0], Abs(res, evaluate=False) ** p,
                         implicit_dims=[iter_dim] + temp_dims)]

            conditional = ConditionalDimension(name='conditional', parent=iter_dim,
                                               condition=(residual[0] >= epsilon),
                                               brk=True)

            # Equation to count, so we can know in what iteration it stopped
            cntEq = Eq(niter[0], conditional, implicit_dims=[time_dim, iter_dim])

            return [tempEq, iterEq] + resEq + [cntEq]

        return []
