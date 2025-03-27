import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae
from dae4py.irk import solve_dae_IRK_generic
from dae4py.bdf import solve_dae_BDF
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
    # t0 = 0.5
    # t1 = 0.7
    # t0 = 0.5
    # t1 = 0.8
    t_span = (t0, t1)

    # initial conditions
    y0, yp0 = true_sol(t0)
    print(f"y0: {y0}")

    # Butcher tableau
    s = 2
    tableau = radau_tableau(s)
    # tableau = gauss_tableau(s)

    # solver settings
    h = 1e-2
    atol = rtol = 1e-6
    sol = solve_dae_IRK_generic(
        F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol, newton_max_iter=10
    )

    y, yp = true_sol(sol.t)
    # print(f"sol.t: {sol.t}")
    # print(f"y(t0): {y[:, 0]}")

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[:, 0], "-k", label="y1")
    ax.plot(sol.t, sol.y[:, 1], "-b", label="y2")
    ax.plot(sol.t, y[0], "ok", label="y1 true")
    ax.plot(sol.t, y[1], "ob", label="y2 true")
    ax.grid()
    ax.legend()
    plt.show()

    exit()

    # h = np.logspace(-1, -8, num=10)
    num = 10
    pow_max = 3
    hs = 10 ** (-np.linspace(1.25, pow_max, num=num))
    print(f"hs: {hs}")

    errors = []
    y_true, yp_true = true_sol(t1)
    for h in hs:
        sol = solve_dae_IRK_generic(
            F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol
        )
        # sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
        t = sol.t
        y = sol.y
        diff = y[-1] - true_sol(t[-1])[0]
        error = np.linalg.norm(diff)
        errors.append(error)
    errors = np.array(errors)

    print(f"errors: {errors}")

    orders = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
    print(f"orders: {orders}")

    fig, ax = plt.subplots()
    ax.loglog(hs, errors, "-o", label="IRK")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 4), "-o", label=f"2 * s - 4 = {2 * s - 4}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 3), "-o", label=f"2 * s - 3 = {2 * s - 3}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 2), "-o", label=f"2 * s - 2 = {2 * s - 2}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 1), "-o", label=f"2 * s - 1 = {2 * s - 1}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s), "-o", label=f"2 * s = {2 * s}")
    ax.grid()
    ax.legend()
    plt.show()
