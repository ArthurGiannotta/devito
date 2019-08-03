from sympy import Function, solve

__all__ = ['JacobiFunction']

#class IterativeFunction(Function):
#    """"""
    #@classmethod
    #def eval(cls, func, lhs, rhs):
    #    return solve(Eq(lhs, rhs), func)

def JacobiFunction(u, lhs, rhs):
    return solve((lhs - rhs).evaluate, u.evaluate)[0]
