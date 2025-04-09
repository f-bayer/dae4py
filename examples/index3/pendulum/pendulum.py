import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau


m = 1
l = 1
g = 10


def F(t, vy, vyp, index=3):
    x, y, u, v, la = vy
    x_dot, y_dot, u_dot, v_dot, la_p = vyp

    R = np.zeros(5, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u
    R[1] = y_dot - v
    R[2] = m * u_dot - 2 * x * la
    R[3] = m * v_dot - 2 * y * la + m * g

    match index:
        case 0:
            R[4] = la_p - (
                m / 2 * (v * g - 2 * (u * u_dot + v * v_dot)) / (x**2 + y**2)
                - m
                / 2
                * (y * g - 2 * (u**2 + v**2))
                * (x * u + y * v)
                / (x**2 + y**2) ** 2
            )

        case 1:
            R[4] = 2 * (x * u_dot + u**2 + y * v_dot + v**2)

        case 2:
            R[4] = 2 * (x * u + y * v)

        case 3:
            R[4] = x**2 + y**2 - 1

    return R


F0 = lambda t, vy, vyp: F(t, vy, vyp, index=0)
F1 = lambda t, vy, vyp: F(t, vy, vyp, index=1)
F2 = lambda t, vy, vyp: F(t, vy, vyp, index=2)
F3 = lambda t, vy, vyp: F(t, vy, vyp, index=3)

if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)

    # initial conditions
    y0_ = np.array([l, 0, 0, 0, 0], dtype=float)
    yp0_ = np.array([0, 0, 0, -g, 0], dtype=float)

    # solver options
    s = 2
    h = 5e-2

    def solve_DAE(F, y0, yp0):
        return solve_dae_IRK(F, y0, yp0, t_span=t_span, h=h, tableau=radau_tableau(s))

    ##############
    # dae solution
    ##############
    sol0 = solve_DAE(F0, y0_, yp0_)
    t0 = sol0.t
    y0 = sol0.y.T
    yp0 = sol0.yp.T

    sol1 = solve_DAE(F1, y0_, yp0_)
    t1 = sol1.t
    y1 = sol1.y.T
    yp1 = sol1.yp.T

    sol2 = solve_DAE(F2, y0_, yp0_)
    t2 = sol2.t
    y2 = sol2.y.T
    yp2 = sol2.yp.T

    sol3 = solve_DAE(F3, y0_, yp0_)
    t3 = sol3.t
    y3 = sol3.y.T
    yp3 = sol3.yp.T

    # export solution
    import sys
    from pathlib import Path

    header = "t_pos, g_pos, t_vel, g_vel, t_acc, g_acc, t_ODE, g_ODE"

    data = np.vstack(
        (
            t3[None, :],
            (y3[0] ** 2 + y3[1] ** 2 - l**2)[None, :],
            t2[None, :],
            (y2[0] ** 2 + y2[1] ** 2 - l**2)[None, :],
            t1[None, :],
            (y1[0] ** 2 + y1[1] ** 2 - l**2)[None, :],
            t0[None, :],
            (y0[0] ** 2 + y0[1] ** 2 - l**2)[None, :],
        )
    ).T

    path = Path(sys.modules["__main__"].__file__)

    np.savetxt(
        path.parent / (path.stem + ".txt"),
        data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t0, y0[0], "-o", label="x ODE.")
    ax[0].plot(t1, y1[0], "-", label="x acc.")
    ax[0].plot(t2, y2[0], "--", label="x vel.")
    ax[0].plot(t3, y3[0], ":", label="x pos.")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t0, y0[1], "-o", label="y ODE.")
    ax[1].plot(t1, y1[1], "-", label="y acc.")
    ax[1].plot(t2, y2[1], "--", label="y vel.")
    ax[1].plot(t3, y3[1], ":", label="y pos.")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t0, y0[0] ** 2 + y0[1] ** 2 - l**2, "-o", label="g ODE.")
    ax[2].plot(t1, y1[0] ** 2 + y1[1] ** 2 - l**2, "-", label="g acc.")
    ax[2].plot(t2, y2[0] ** 2 + y2[1] ** 2 - l**2, "--", label="g vel.")
    ax[2].plot(t3, y3[0] ** 2 + y3[1] ** 2 - l**2, ":", label="g pos.")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t1, 2 * y0[0] * y0[2] + 2 * y0[1] * y0[3], "-o", label="g_dot ODE.")
    ax[3].plot(t1, 2 * y1[0] * y1[2] + 2 * y1[1] * y1[3], "-", label="g_dot acc.")
    ax[3].plot(t2, 2 * y2[0] * y2[2] + 2 * y2[1] * y2[3], "--", label="g_dot vel.")
    ax[3].plot(t3, 2 * y3[0] * y3[2] + 2 * y3[1] * y3[3], ":", label="g_dot pos.")
    ax[3].legend()
    ax[3].grid()

    plt.show()
