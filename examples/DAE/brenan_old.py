import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau


"""Index 1 DAE found in Chapter 4 of Brenan1996.

References:
-----------
Brenan1996: https://doi.org/10.1137/1.9781611971224.ch4
"""


def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - t * y2p + y1 - (1 + t) * y2
    # TODO: Derive analytical solution for this problem
    # F[0] = y1p - (1 / t) * y2p + y1 - (1 + 1 / t) * y2
    F[1] = y2 - np.sin(t)

    return F


def true_sol(t):
    return (
        np.array(
            [
                np.exp(-t) + t * np.sin(t),
                np.sin(t),
            ]
        ),
        np.array(
            [
                -np.exp(-t) + np.sin(t) + t * np.cos(t),
                np.cos(t),
            ]
        ),
    )


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 2e1
    t_span = (t0, t1)

    # initial conditions
    y0, yp0 = true_sol(t0)

    # Butcher tableau
    s = 2
    tableau = radau_tableau(s)
    # tableau = gauss_tableau(s)

    # solver options
    h = 1e-3
    atol = rtol = 1e-6
    sol = solve_dae_IRK_generic(F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    h = sol.h

    # visualization
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[:, 0], "--or", label="y1")
    ax[0].plot(t, y[:, 1], "--og", label="y2")

    y_true, yp_true = true_sol(t)
    ax[0].plot(t, y_true[0], "-r", label="y1 true")
    ax[0].plot(t, y_true[1], "-g", label="y2 true")

    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, h, "-o", label="h")
    ax[1].grid()
    ax[1].legend()

    plt.show()
