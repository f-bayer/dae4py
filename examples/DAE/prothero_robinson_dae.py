import numpy as np
import matplotlib.pyplot as plt
from dae4py.dae_problem import DAEProblem
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.benchmark import convergence_analysis


omega = 3
# eps = 1e3
eps = 1e1
la = -20

phi1 = lambda t: np.arctan(eps * np.cos(omega * t))
phi2 = lambda t: np.sin(t)

phi1_dot = (
    lambda t: -eps * omega * np.sin(omega * t) / (1 + (eps * np.cos(omega * t)) ** 2)
)
phi2_dot = lambda t: np.cos(t)


def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - la * (y1 - phi1(t) * y2) - phi1_dot(t) * y2 - phi1(t) * y2p
    F[1] = y2 - phi2(t)

    return F


def true_sol(t):
    return (
        np.array(
            [
                phi1(t) * phi2(t),
                phi2(t),
            ]
        ),
        np.array(
            [
                phi1_dot(t) * phi2(t) + phi1(t) * phi2_dot(t),
                phi2_dot(t),
            ]
        ),
    )


problem = DAEProblem(
    name="Extended Prothero-Robinson problem",
    F=F,
    # t_span=(-0.52, 0.52),
    # t_span=(-2, 2),
    t_span=(-1, 1),
    index=1,
    true_sol=true_sol,
)


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-3
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
    ax[0].plot(t, y[:, 1], "--og", label="y2")

    y_true, yp_true = true_sol(t)
    ax[0].plot(t, y_true[0], "-r", label="y1 true")
    ax[0].plot(t, y_true[1], "-g", label="y2 true")

    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, h, "-o", label="h")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_yscale("log")

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    pow_max = 9
    h_max = Dt / 4
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
