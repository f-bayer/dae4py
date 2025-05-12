import numpy as np
from dae4py.dae_problem import DAEProblem

# Modulating function and its derivative
def f(yp):
    return -yp^2

def f_prime(yp):
    return -2*yp

def f_prime_inv(s):
    # Inverse function of f_prime. Required for true singular solution
    return -0.5*s

def f_prime_inv_int(s):
    # Integral of f_prime. Required for true singular solution
    return -0.25*s**2

class ClairautDAEProblem(DAEProblem):
    def __init__(self, t_span, C=None):

        t0 = t_span[0]

        """Determine initial condition based on value of C"""
        if C is None:
            # Singular solution (envelope): 
            # t + f'(yp) === 0
            yp0 = f_prime_inv(-t0)
        else:
            # General solution (line):
            # ypp === 0
            yp0 = C

        y0 = t0*yp0 + f(yp0)

        
        super().__init__(
            name="Clairaut", 
            F=None, 
            t_span=t_span, 
            index=0, 
            y0=y0, 
            yp0=yp0, 
            true_sol=None,
        )

    # implicit differential equation
    def F(self, t, y, yp):
        return -y + t*yp + f(yp)
    
    # setter is required to handle self.F = F in super().__init__
    @F.setter 
    def F(self, value):
        if value is not None:
            raise AttributeError('F cannot be set explicitly in Clairaut example. Change f instead and instantiate new object.')
        
    def true_sol(self, t):
        if self.C is None:
            # Singular solution (envelope)
            yp = f_prime_inv(-t)
            y = self.y0 + f_prime_inv_int(-self.t_span[0]) - f_prime_inv_int(-t)

        else:
            # General solution (straight lines)
            y0 = self.y0

    @true_sol.setter
    def true_sol(self, value):
        if value is not None:
            raise AttributeError('true_sol cannot be set explicitly in Clairaut example. Change f instead and instantiate new object.')