from devito.equation import Eq
from devito.types import Dimension, TimeFunction
from enum import Enum, auto
from sympy import solve

__all__ = ['IterativeMethod']

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class IterativeMethod(Enum):
    GAUSS_SEIDEL = auto()
    SOR = auto()
    SUCCESSIVE_OVER_RELAXATION = SOR

    @static_vars(constants=dict())
    def set_constant(self, name, value):
        IterativeMethod.set_constant.constants[name] = value

    @static_vars(temporaries=[])
    def explicit_expressions(self, equation):
        """
        Get the explicit expressions equivalent to an implicit equation.

        Parameters
        ----------
        equation : Eq
            The implicit equation to be processed.
        """

        function = next(iter(equation.lhs._functions))
        temp_index = str(len(IterativeMethod.explicit_expressions.temporaries))

        # Dimension in which to execute the iterations
        iter_dim = Dimension(name='iter')

        # Create a new set of "temporary" spatial dimensions
        temp_dims = [function.dimensions[0]]
        orig_to_temp_subs = []
        temp_to_orig_subs = []

        for dimension in function.dimensions[1:]:
            temp_dimension = Dimension('tempdim_' + dimension.name + temp_index)
            temp_dims += [temp_dimension]

            orig_to_temp_subs += [(dimension, temp_dimension)]
            temp_to_orig_subs += [(temp_dimension, dimension)]

        if self.value == self.GAUSS_SEIDEL.value or self.value == self.SOR.value:
            temp = TimeFunction(name = 'temp_' + temp_index, shape = function.shape, 
                                dimensions = temp_dims)
            IterativeMethod.explicit_expressions.temporaries += [temp]

            temp_subs = temp.subs(temp_to_orig_subs)
            gauss_seidel = solve((equation.lhs - temp_subs).evaluate, function.forward,
                                rational = False, simplify = False)[0]

        if self.value == self.GAUSS_SEIDEL.value:
            # This expression calculates the b coefficients
            expr1 = Eq(temp_subs, equation.rhs)

            # This is the Gauss Seidel expression
            expr2 = Eq(function.forward.subs(orig_to_temp_subs),
                gauss_seidel.subs(orig_to_temp_subs), implicit_dims=[iter_dim]+temp_dims[1:])

            return [expr1, expr2]
        elif self.value == self.SOR.value:
            omega = IterativeMethod.set_constant.constants.get('omega', 1.2)
            sor = omega * gauss_seidel + (1 - omega) * function.forward

            # This expression calculates the b coefficients
            expr1 = Eq(temp_subs, equation.rhs)

            # This is the Gauss Seidel expression
            expr2 = Eq(function.forward.subs(orig_to_temp_subs),
                sor.subs(orig_to_temp_subs), implicit_dims=[iter_dim]+temp_dims[1:])

            return [expr1, expr2]

        return []

#class IterativeFunction(Function):
#    """"""
    #@classmethod
    #def eval(cls, func, lhs, rhs):
    #    return solve(Eq(lhs, rhs), func)
