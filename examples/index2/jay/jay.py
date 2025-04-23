import numpy as np
from dae4py.dae_problem import DAEProblem


def F(t, y, yp, nonlinear_multiplier=False):
    y1, y2, la = y
    y1p, y2p, _ = yp

    F = np.zeros(3, dtype=y.dtype)
    if nonlinear_multiplier:
        F[0] = y1p - (y1 * y2**2 * la**2)
    else:
        F[0] = y1p - (y1 * y2 * la)
    F[1] = y2p - (y1**2 * y2**2 - 3 * y2**2 * la)
    F[2] = y1**2 * y2 - 1.0

    return F


def true_sol(t):
    y = np.array(
        [
            np.exp(t),
            np.exp(-2 * t),
            np.exp(2 * t),
        ]
    )

    yp = np.array(
        [
            np.exp(t),
            -2 * np.exp(-2 * t),
            2 * np.exp(2 * t),
        ]
    )

    return y, yp


problem = DAEProblem(
    name="Jay",
    F=F,
    t_span=(0, 1),
    index=2,
    true_sol=true_sol,
)
