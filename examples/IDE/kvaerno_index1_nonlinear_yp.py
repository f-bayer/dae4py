import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau


"""Nonlinear index 1 DAE, see P4 in Kvaerno1990.

References:
-----------
Kvaerno1990: https://doi.org/10.2307/2008502
"""


def F(t, y, yp):
    y1, y2 = y
    yp1, yp2 = yp
    return np.array(
        [
            (np.sin(yp1) ** 2 + np.cos(y2) ** 2) * yp2**2
            - (t - 6) ** 2 * (t - 2) ** 2 * y1 * np.exp(-t),
            (4 - t) * (y2 + y1) ** 3 - 64 * t**2 * np.exp(-t) * y1 * y2,
        ]
    )


def true_sol(t):
    return (
        np.array(
            [
                t**4 * np.exp(-t),
                (4 - t) * t**3 * np.exp(-t),
            ]
        ),
        np.array(
            [
                (4 * t**3 - t**4) * np.exp(-t),
                ((4 - t) * 3 * t**2 - (5 - t) * t**3) * np.exp(-t),
            ]
        ),
    )


if __name__ == "__main__":
    # time span
    t0 = 0.1
    t1 = 1.2
    t_span = (t0, t1)

    # initial conditions
    y0, yp0 = true_sol(t0)
    print(f"y0: {y0}")

    # Butcher tableau
    s = 2
    tableau = radau_tableau(s)
    # tableau = gauss_legendre_tableau(s)

    # solver settings
    h = 1e-2
    atol = rtol = 1e-6
    sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol)

    y, yp = true_sol(sol.t)

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[:, 0], "-k", label="y1")
    ax.plot(sol.t, sol.y[:, 1], "-b", label="y2")
    ax.plot(sol.t, y[0], "ok", label="y1 true")
    ax.plot(sol.t, y[1], "ob", label="y2 true")
    ax.grid()
    ax.legend()
    plt.show()
