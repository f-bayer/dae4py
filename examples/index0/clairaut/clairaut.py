import numpy as np
from typing import override
from dae4py.dae_problem import DAEProblem

CASES = ["quadratic_neg", "quadratic_pos", "cubic_neg", "cubic_pos", "ln", "sqrt", "generalized"]
CASE = CASES[0] 


# Modulating function and its derivative
match CASE:
    case "quadratic_neg":
        T_SPAN = [-10, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return -yp**2

        def f_prime(yp):
            return -2*yp
        
        def f_prime_prime(yp):
            return -2

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
        T_SPAN = [0, 300]
        C_SPAN = [-10, 10]

        def f(yp):
            return -yp**3

        def f_prime(yp):
            return -3*yp**2

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            if np.all(s <= 0):
                return np.sqrt(-(1/3)*s)
            else:
                raise ValueError(f"case '{CASE}': {s} not in the image of f_prime. Adjust T_SPAN.")

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            if np.all(lower <= 0) and np.all(upper <= 0):
                return 2*(-(1/3)*lower)**(1.5) - 2*(-(1/3)*upper)**(1.5)
            else:
                raise ValueError(f"case '{CASE}': {lower} or {upper} not in the image of f_prime. Adjust T_SPAN.")
            
    case "cubic_pos":
        T_SPAN = [-300, -1]
        C_SPAN = [-10, 10]

        def f(yp):
            return 2*yp**3

        def f_prime(yp):
            return 6*yp**2

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution
            if np.all(s >= 0):
                return np.sqrt((1/6)*s)
            else:
                raise ValueError(f"case '{CASE}': {s} not in the image of f_prime. Adjust T_SPAN.")

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            if np.all(lower >= 0) and np.all(upper >= 0):
                return 4*((1/6)*upper)**(1.5) - 4*(1/6*lower)**(1.5)
            else:
                raise ValueError(f"case '{CASE}': {lower} or {upper} not in the image of f_prime. Adjust T_SPAN.")
            
    case "ln":
        T_SPAN = [-10, 0]
        C_SPAN = [-20000, 20000]
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
        T_SPAN = [-10, 0]
        # C_SPAN = [-300, 300]
        C_SPAN = [-100, 100]

        def f(yp):
            return (2/3)* yp * np.sqrt(np.abs(yp))

        def f_prime(yp):
            return np.sqrt(np.abs(yp))

        def f_prime_inv(s):
            # Inverse function of f_prime. Required for true singular solution.
            # Positive branch.
            return s ** 2

        def f_prime_inv_int(lower, upper):
            # Integral of f_prime_inv from lower to upper.
            # Required for true singular solution.
            return 1/3 * (upper ** 3 - lower ** 3)

    case _:
        raise ValueError(f"Clairaut case {CASE} unknown. Allowed cases: {CASES}")

class ClairautDAEProblem(DAEProblem):
    def __init__(self, C=None, index=0):

        t0 = T_SPAN[0]
        self.C = C

        """Determine initial condition based on value of C"""
        if self.C is None:
            # Singular solution (envelope): 
            # t + f'(yp) === 0
            yp0 = np.atleast_1d(f_prime_inv(-t0))
            index = index
        else:
            if index > 0:
                raise ValueError('Only singular solution may have index=1')
            # General solution (line):
            # ypp === 0
            yp0 = np.atleast_1d(C)
            index = 0

        y0 = t0*yp0 + f(yp0)

        if index > 0:
            y0 = np.append(y0, 0)
            yp0 = np.append(yp0, 0)

        super().__init__(
            name="Clairaut", 
            F=self.F, 
            t_span=T_SPAN, 
            index=index, 
            y0=y0, 
            yp0=yp0, 
            true_sol=self.true_sol,
            # jac=self.jac
        )

    # implicit differential equation
    def F(self, t, y, yp):

        if self.index > 0:
            mu = y[1, ...]
            y = y[0, ...]
            mup = yp[1, ...]
            yp = yp[0, ...]

            constr = t + f_prime(yp)
            dconstr_dyp = f_prime_prime(yp)

        r = np.atleast_1d(-y + t*yp + f(yp))

        if self.index > 0:
            # Use Gear's idea to append a constraint to remain on the singularity
            r = r + dconstr_dyp * mup
            r = np.hstack((r, constr))
        
        return r
    
    # def jac(self, t, y, yp):
    #     if self.idx > 0:
    #         mu = y[1, ...]
    #         y = y[0, ...]
    #         mup = yp[1, ...]
    #         yp = yp[0, ...]

    #         drdmu = np.zeros_like(mup)
    #         drdmup = f_prime_prime(yp)

    #         dconstr_dyp = f_prime_prime(yp)
    #         dconstr_dmu = np.zeros_like(mup)
    #         dconstr_dmup = np.zeros_like(mup)


    #     drdy = np.ones_like(y)
    #     drdyp = t*np.ones_like(yp) + f_prime(yp)

    #     if self.idx > 0:
    #         drdyp = drdyp + mup*f_prime_prime_prime(yp)
    #     return drdyp, drdy
        
    
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
    
class ClairautYDAEProblem(ClairautDAEProblem):
    def __init__(self, C=None):

        if CASE != "cubic_pos":
            raise ValueError('YDAE only works for case cubic_pos!')
        
        super().__init__(C)

        self.u0 = self.y0
        self.up0 = self.yp0

    @override
    def F(self, t, y, yp):
        u = y ** 2
        up = 2*y*yp
        return super().F(t, u, up)
    
    @override
    def true_sol(self, t):
        u, up = super().true_sol(t)
        y = np.sqrt(u)
        yp = 0.5*up/y
        return y, yp
    
"""Check consistency"""
h = 1e-4

Cs = np.linspace(*C_SPAN, 100)
fs = f(Cs)

f_primes = f_prime(Cs)
f_primes_approx = 1/(2*h)*(f(Cs + h) - f(Cs - h))
print(f"Clairaut '{CASE}'")
print(f"Max error f finite difference: {np.max(np.abs(f_primes - f_primes_approx))}")

ts = -np.linspace(*T_SPAN, 100)
f_prime_invs = f_prime_inv(ts)
ts_reconstructed = f_prime(f_prime_invs)
print(f"Max inversion error: {np.max(np.abs(ts - ts_reconstructed))}")


f_prime_invs_approx = 1/(2*h)*(f_prime_inv_int(lower=ts[0], upper=ts[1:-1] + h) - f_prime_inv_int(lower=ts[0], upper=ts[1:-1] - h))
print(f"Max error f_inv finite difference: {np.max(np.abs(f_prime_invs[1:-1] - f_prime_invs_approx))}")


