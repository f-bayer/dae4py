import numpy as np
from scipy.special import binom, factorial
import matplotlib.pyplot as plt
from dae4py.butcher_tableau import gauss_legendre_tableau, radau_tableau


def P(k, j, z):
    return np.sum(
        [
            binom(k, i)
            * z**i
            * factorial(k + j + i, exact=True)
            / factorial(k + j, exact=True)
            for i in range(k)
        ]
    )


def Q(k, j, z):
    return P(j, k, -z)


def R(k, j, z):
    return P(k, j, z) / Q(k, j, z)


def stability_domains(tableau, xrange=[-30, 30], yrange=[-30, 30], nx=400, ny=400):
    A = tableau.A
    b = tableau.b
    s = tableau.s

    x = np.linspace(*xrange, num=nx)
    y = np.linspace(*yrange, num=ny)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    I = np.ones(s)
    E = np.eye(s)
    R = np.zeros_like(Z, dtype=complex)
    for i in range(nx):
        for j in range(ny):
            R[i, j] = 1 + Z[i, j] * b.T @ np.linalg.solve(E - Z[i, j] * A, I)

    abs_R = np.abs(R)

    return X, Y, Z, abs_R


if __name__ == "__main__":
    smin = 1
    smax = 5
    ss = np.arange(smin, smax + 1)

    fig, ax = plt.subplots(1, 2)

    # gauss = []
    for s in ss:
        X, Y, Z, abs_R = stability_domains(gauss_legendre_tableau(s))

        contour = ax[0].contour(X, Y, abs_R, levels=[1], colors="black")
        ax[0].set_xlabel("Re(z)")
        ax[0].set_ylabel("Im(z)")
        ax[0].set_title("Gauss-Legendre")
        ax[0].grid()
        ax[0].axis("equal")

        vertices = contour.collections[0].get_paths()[0].vertices
        # gauss.append(vertices)

        np.savetxt(
            f"stability_domain_gauss_legendre_s{s}.txt",
            vertices,
            header="x, y",
            delimiter=", ",
            comments="",
        )

    # radau = []
    for s in ss:
        X, Y, Z, abs_R = stability_domains(radau_tableau(s))

        contour = ax[1].contour(X, Y, abs_R, levels=[1], colors="black")
        ax[1].set_xlabel("Re(z)")
        ax[1].set_ylabel("Im(z)")
        ax[1].set_title("Radau IIA")
        ax[1].grid()
        ax[1].axis("equal")

        vertices = contour.collections[0].get_paths()[0].vertices
        # radau.append(vertices)

        np.savetxt(
            f"stability_domain_radau_s{s}.txt",
            vertices,
            header="x, y",
            delimiter=", ",
            comments="",
        )

    # header = "".join([f"x{s}, y{s}, " for s in ss])[:-2]
    # data = np.concatenate([gi.T for gi in gauss])
    # np.savetxt(
    #     "stability_domain_gauss_legendre.txt",
    #     data,
    #     header=header,
    #     delimiter=", ",
    #     comments="",
    # )

    # data = np.concatenate([ri.T for ri in radau])
    # np.savetxt(
    #     "stability_domain_radauIIA",
    #     data,
    #     header=header,
    #     delimiter=", ",
    #     comments="",
    # )

    plt.show()

    # np.savetxt(
    #     "implicit_Euler_h1.txt",
    #     np.array([t1, *y1.T]).T,
    #     header="t, y1",
    #     delimiter=", ",
    #     comments="",
    # )

    # np.savetxt(
    #     "trapezoidal_h1.txt",
    #     np.array([t2, *y2.T]).T,
    #     header="t, y1",
    #     delimiter=", ",
    #     comments="",
    # )

    # np.savetxt(
    #     "trapezoidal_h2.txt",
    #     np.array([t3, *y3.T]).T,
    #     header="t, y1",
    #     delimiter=", ",
    #     comments="",
    # )

    # fig, ax = plt.subplots()
    # ax.plot(t1, y1, "-ok", label="impl. Euler (h = 1.5 / 40)")
    # ax.plot(t2, y2, "--xr", label="trapezoidal rule (h = 1.5 / 40)")
    # ax.plot(t3, y3, "-.sb", label="trapezoidal rule (h = 1.5 / 80)")
    # ax.grid()
    # ax.legend()
    # plt.show()
