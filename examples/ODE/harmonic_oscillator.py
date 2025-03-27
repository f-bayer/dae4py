import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK_generic
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau


omega = 1e1


def F(t, y, yp):
    x, v = y
    xp, vp = yp
    return np.array([xp - v, vp + omega**2 * x])


def true_sol(t):
    y0 = np.array([1.0, 0.0])
    x0, v0 = y0
    return (
        np.array(
            [
                x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t),
                -omega * x0 * np.sin(omega * t) + v0 * np.cos(omega * t),
            ]
        ),
        np.array(
            [
                -omega * x0 * np.sin(omega * t) + v0 * np.cos(omega * t),
                -(omega**2) * x0 * np.cos(omega * t) - omega * v0 * np.sin(omega * t),
            ]
        ),
    )


if __name__ == "__main__":
    # time span
    t0 = 0.0
    t1 = 2 * np.pi * omega * 1e-2
    t_span = (t0, t1)

    y0, yp0 = true_sol(t0)

    # initial conditions
    y0, yp0 = true_sol(t0)

    # step-sizes
    num = 10
    pow_max = 3
    hs = 10 ** (-np.linspace(2, pow_max, num=num))
    print(f"hs: {hs}")

    # Radau method for now
    s = 3
    tableau = radau_tableau(s)
    # tableau = gauss_tableau(s)

    # solver options
    atol = rtol = 1e-12

    errors = []
    y_true, yp_true = true_sol(t1)
    for h in hs:
        sol = solve_dae_IRK_generic(
            F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol, newton_max_iter=20
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
    ax.loglog(hs, errors, "-o", label="Radau IIA")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 4), "-o", label=f"2 * s - 4 = {2 * s - 4}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 3), "-o", label=f"2 * s - 3 = {2 * s - 3}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 2), "-o", label=f"2 * s - 2 = {2 * s - 2}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s - 1), "-o", label=f"2 * s - 1 = {2 * s - 1}")
    ax.loglog(hs, 1e-1 * hs ** (2 * s), "-o", label=f"2 * s = {2 * s}")
    ax.grid()
    ax.legend()
    plt.show()
