import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from particle_circular_track import problem


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-3
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(problem.F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(
            problem.F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol
        )
    t = sol.t
    y = sol.y

    # visualization
    y_true, yp_true = problem.true_sol(t)

    n = len(y0)
    fig, ax = plt.subplots(n, 1)
    for i in range(n):
        ax[i].plot(t, y[:, i], "-r", label=f"y{i}")
        ax[i].plot(t, y_true[i], "x", label=f"y{i} true")
        ax[i].grid()
        ax[i].legend()

    plt.show()


if __name__ == "__main__":
    trajectory()  # BDF case
    trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)
