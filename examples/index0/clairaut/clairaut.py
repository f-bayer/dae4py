import numpy as np
from dae4py.dae_problem import DAEProblem

CASES = ["quadratic_neg", "quadratic_pos", "cubic_neg", "ln", "sqrt"]
CASE = CASES[3] 


# Modulating function and its derivative
match CASE:
    case "quadratic_neg":
        T_SPAN = [-10, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return -yp**2

        def f_prime(yp):
            return -2*yp

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            return -0.5*s

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution
            return -0.25*upper**2 + 0.25*lower**2
        
    case "quadratic_pos":
        T_SPAN = [-10, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return yp**2

        def f_prime(yp):
            return 2*yp

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            return 0.5*s

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution
            return 0.25*upper**2 - 0.25*lower**2
        
    case "cubic_neg":
        T_SPAN = [0, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return -yp**3

        def f_prime(yp):
            return -3*yp**2

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            if np.all(s <= 0):
                return np.sqrt(-(1/3)*s)
            else:
                raise ValueError(f"case '{CASE}': {s} not in the image of f_prime")

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            if np.all(lower <= 0) and np.all(upper <= 0):
                return -2*(-(1/3)*lower)**(1.5) + 2*(-(1/3)*upper)**(1.5)
            else:
                raise ValueError(f"case '{CASE}': {lower} or {upper} not in the image of f_prime")
            
    case "ln":
        T_SPAN = [-10, 0]
        C_SPAN = [-1000, 1000]
        def f(yp):
                return (np.log(np.abs(yp))-1)*yp


        def f_prime(yp):
            return np.log(np.abs(yp))

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution.
            # We only consider the positive branch...
            return np.exp(s)

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            return np.exp(upper) - np.exp(lower)

    case "sqrt":
        T_SPAN = [0.01, 10]
        def f(yp):
            if np.all(yp > 0):
                return np.sqrt(yp)
            else:
                raise ValueError(f"case '{CASE}': sqrt({yp}) not well defined")

        def f_prime(yp):
            return np.log(yp)

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            return np.exp(s)

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            return np.exp(upper) - np.exp(lower)

    case _:
        raise ValueError(f"Clairaut case {CASE} unknown. Allowed cases: {CASES}")

class ClairautDAEProblem(DAEProblem):
    def __init__(self, C=None):

        t0 = T_SPAN[0]
        self.C = C

        """Determine initial condition based on value of C"""
        if self.C is None:
            # Singular solution (envelope): 
            # t + f'(yp) === 0
            yp0 = np.atleast_1d(f_prime_inv(-t0))
        else:
            # General solution (line):
            # ypp === 0
            yp0 = np.atleast_1d(C)

        y0 = t0*yp0 + f(yp0)

        super().__init__(
            name="Clairaut", 
            F=self.F, 
            t_span=T_SPAN, 
            index=0, 
            y0=y0, 
            yp0=yp0, 
            true_sol=self.true_sol,
        )

    # implicit differential equation
    def F(self, t, y, yp):
        return -y + t*yp + f(yp)
    
    def true_sol(self, t):
        if self.C is None:
            # Singular solution (envelope)
            yp = f_prime_inv(-t)
            y = self.y0 + f_prime_inv_int(lower=-t, upper=-self.t_span[0])

        else:
            # General solution (straight lines)
            yp = self.C*np.ones_like(np.atleast_1d(t))
            y = self.C*t + f(self.yp0)

        return y, yp