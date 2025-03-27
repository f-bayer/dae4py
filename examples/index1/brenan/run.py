import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from brenan import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-1
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    y_true, yp_true = problem.true_sol(t)
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"y1")
    ax[0, 0].plot(t, y_true[0], "rx", label=f"y1 true")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"y2")
    ax[1, 0].plot(t, y_true[1], "rx", label=f"y2 true")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"yp1")
    ax[0, 1].plot(t, yp_true[0], "rx", label=f"yp1 true")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"yp2")
    ax[1, 1].plot(t, yp_true[1], "rx", label=f"yp2 true")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    trajectory()  # BDF case
    trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)
