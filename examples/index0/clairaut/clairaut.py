import numpy as np
from dae4py.dae_problem import DAEProblem

# Modulating function
def f(up):
    return np.ln(up) # ensure df/dup >= 0 and df/dup(0) != 0

# implicit differential equation
def F(t, y, yp):
    return -y + t*yp + f(yp)

