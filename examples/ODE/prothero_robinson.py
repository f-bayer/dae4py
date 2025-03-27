import numpy as np
import matplotlib.pyplot as plt
from dae4py.dae_problem import DAEProblem
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.benchmark import convergence_analysis


la = -20
phi = lambda t: np.arctan(2 * t)
phi_dot = lambda t: 2 / (1 + 4 * t**2)


def F(t, y, yp):
    return yp - la * (y - phi(t)) - phi_dot(t)


def true_sol(t):
    return (
        np.atleast_1d(phi(t)),
        np.atleast_1d(phi_dot(t)),
    )


problem = DAEProblem(
    name="Prothero-Robinson problem",
    F=F,
    t_span=(-1, 1),
    index=0,
    true_sol=true_sol,
)


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-1
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK_generic(
            F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol
        )
    t = sol.t
    y = sol.y
    h = sol.h

    # visualization
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[:, 0], "--or", label="y1")

    y_true, yp_true = true_sol(t)
    ax[0].plot(t, y_true, "-r", label="y1 true")

    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, h, "-o", label="h")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_yscale("log")

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    pow_max = 8
    h_max = Dt / 2
    h0s = h_max * (1 / 2) ** (np.arange(pow_max, dtype=float))

    rtols = 1e-16 * np.ones_like(h0s)
    atols = 1e-16 * np.ones_like(h0s)

    print(f"rtols: {rtols}")
    print(f"atols: {atols}")
    print(f"h0s: {h0s}")

    convergence_analysis(
        problem,
        rtols,
        atols,
        h0s,
    )


if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    # trajectory(s=2, tableau=radau_tableau)

    convergence()
