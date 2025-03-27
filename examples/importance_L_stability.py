import numpy as np
from scipy._lib._util import _RichResult
from dae4py.irk import solve_dae_IRK
import matplotlib.pyplot as plt


def explicit_euler_tableau():
    A = np.zeros((1, 1), dtype=float)
    b = np.ones(1, dtype=float)
    c = np.sum(A, axis=1)
    p = 1
    q = 1
    s = 1
    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)


def implicit_euler_tableau():
    A = np.ones((1, 1), dtype=float)
    b = A[-1]
    c = np.sum(A, axis=1)
    p = 1
    q = 1
    s = 1
    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)


def trapezoidal_rule_tableau():
    A = np.array(
        [
            [0, 0],
            [1 / 2, 1 / 2],
        ],
        dtype=float,
    )
    b = A[-1]
    c = np.sum(A, axis=1)
    p = 2
    q = 2
    s = 2
    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)


def theta_method_tableau(theta=0.4):
    A = np.array(
        [
            [0, 0],
            [1 - theta, theta],
        ],
        dtype=float,
    )
    b = A[-1]
    c = np.sum(A, axis=1)
    p = 1
    q = 1
    s = 2
    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)


def f(t, y):
    return -2000 * (y - np.cos(t))


def F(t, y, yp):
    return yp - f(t, y)


if __name__ == "__main__":
    tableau1 = implicit_euler_tableau()
    tableau2 = trapezoidal_rule_tableau()
    tableau3 = explicit_euler_tableau()
    # tableau3 = theta_method_tableau()
    print(f"tableau1:\n{tableau1}")
    print(f"tableau2:\n{tableau2}")
    print(f"tableau3:\n{tableau3}")

    t0 = 0
    t1 = 1.5
    t_span = (t0, t1)

    h1 = 1.5 / 40
    h2 = 1.5 / 80
    h3 = 1.5 / 1500

    y0 = np.zeros(1, dtype=float)
    yp0 = f(t0, y0)

    sol1 = solve_dae_IRK(
        F,
        y0,
        yp0,
        t_span,
        h1,
        tableau1,
    )
    t1 = sol1.t
    y1 = sol1.y

    sol2 = solve_dae_IRK(
        F,
        y0,
        yp0,
        t_span,
        h1,
        tableau2,
    )
    t2 = sol2.t
    y2 = sol2.y

    sol3 = solve_dae_IRK(
        F,
        y0,
        yp0,
        t_span,
        h2,
        tableau2,
    )
    t3 = sol3.t
    y3 = sol3.y

    sol4 = solve_dae_IRK(
        F,
        y0,
        yp0,
        t_span,
        h3,
        tableau3,
    )
    t4 = sol4.t
    y4 = sol4.y

    np.savetxt(
        "implicit_Euler_h1.txt",
        np.array([t1, *y1.T]).T,
        header="t, y1",
        delimiter=", ",
        comments="",
    )

    np.savetxt(
        "trapezoidal_h1.txt",
        np.array([t2, *y2.T]).T,
        header="t, y1",
        delimiter=", ",
        comments="",
    )

    np.savetxt(
        "trapezoidal_h2.txt",
        np.array([t3, *y3.T]).T,
        header="t, y1",
        delimiter=", ",
        comments="",
    )

    np.savetxt(
        "explicit_Euler_h3.txt",
        np.array([t4, *y4.T]).T,
        header="t, y1",
        delimiter=", ",
        comments="",
    )

    fig, ax = plt.subplots()
    ax.plot(t1, y1, "-ok", label="impl. Euler (h = 1.5 / 40)")
    # ax.plot(t2, y2, "--xr", label="trapezoidal rule (h = 1.5 / 40)")
    # ax.plot(t3, y3, "-.sb", label="trapezoidal rule (h = 1.5 / 80)")
    ax.plot(t4, y4, ":g", label="expl. Euler (h = 1.5 / 2000)")
    ax.grid()
    ax.legend()
    ax.set_ylim(-0.1, 1.75)
    plt.show()
