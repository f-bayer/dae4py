import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau
from clairaut import ClairautYDAEProblem, C_SPAN

def trajectory(C, s=None, tableau=None, axs=None):
    problem = ClairautYDAEProblem(C)
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-2
    atol = rtol = 1e-6
    # if s is None or tableau is None:
    #     sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    # else:
    sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # export solution
    np.savetxt(
        "clairaut.txt",
        np.hstack([t[:, None], y, yp]),
        header="t, y, yp",
        delimiter=", ",
        comments="",
    )

    # visualization
    y_true, _ = problem.true_sol(t)

    if axs is None:
        fig, axs = plt.subplots(1)
        axs = [axs] # brackets are only needed if argument of plt.subplots is 1

    if C is None:
        col_true = 'green'
        col_int = 'green'
        width=3
    else:
        col_true='r'
        col_int = 'k'
        width=1
    
    axs[0].plot(t, y, "-", label=f"y", color=col_int, linewidth=width)
    axs[0].plot(t, y_true, "--", label=f"y true", linewidth=width, color=col_true)

    # axs[1].plot(t, yp, "-k", label=f"yp")
    # axs[1].plot(t, yp_true, "rx", label=f"yp true")
    # axs[1].grid()
    # axs[1].legend()

    return axs


if __name__ == '__main__':
    axs = None
    # General solutions
    for C in np.linspace(*C_SPAN, 20):
        axs = trajectory(C, s=2, tableau=radau_tableau, axs=axs)

    # Singular solution
    axs = trajectory(None, s=2, tableau=radau_tableau, axs=axs)
    plt.show()