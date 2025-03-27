import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau


# mu = 1e1
# mu = 5
# eps = 1 / mu**2

eps = 1e-3
T = 5

# eps = 1e-6
# T = 4

mu = 1 / np.sqrt(eps)
# T = (3 - 2 * np.log(2)) * mu


def rhs(t, y):
    z1, z2 = y
    return np.array(
        [
            z2,
            ((1 - z1**2) * z2 - z1) / eps,
        ]
    )


def F(t, y, yp):
    # return yp - rhs(t, y)
    z1, z2 = y
    zp1, zp2 = yp
    return np.array([zp1 - z2, eps * zp2 - (1 - z1**2) * z2 + z1])

    # zp1 = z2 = 2
    # z1 = (1 - z1**2) * z2 = z2 - z2 * z1**2
    # z1 + z2 * z1**2 = z2
    # z2 * z1**2 + z1 - z2
    # z1 = (-1 +- np.sqrt(1 - 4 * z2 * (-z2))) / (2 * z2)


if __name__ == "__main__":
    # time span
    t0 = 0.0
    t1 = t0 + T
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([2, 0], dtype=float)
    yp0 = rhs(t0, y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")

    # z2 = 2
    # z1p = z2
    # z2p = 0
    # z1 = (-1 - np.sqrt(1 - 4 * z2 * (-z2))) / (2 * z2)

    # y0 = np.array([z1, z2], dtype=float)
    # yp0 = np.array([z1p, z2p], dtype=float)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"F(t0, y0, yp0): {F(t0, y0, yp0)}")
    # # exit()

    # Radau method for now
    s = 2
    tableau = radau_tableau(s)
    # tableau = gauss_tableau(s)

    # solver options
    h = 1e-4
    atol = rtol = 1e-4
    # sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol)
    sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    h = sol.h

    # export
    np.savetxt(
        "RadauIIA(2)_van_der_pol.txt",
        np.array([t, *y.T, h]).T,
        header="t, z1, z2, h",
        delimiter=", ",
        comments="",
    )

    # visualization
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, y[:, 0], "-o", label="z_1")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, y[:, 1], "-o", label="z_2")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, h, "-o", label="h")
    ax[2].grid()
    ax[2].legend()

    plt.show()
