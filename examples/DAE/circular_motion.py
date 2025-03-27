import numpy as np
import matplotlib.pyplot as plt
from dae4py.dae_problem import DAEProblem
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.benchmark import convergence_analysis

from dae4py.irk import solve_dae_IRK


INDEX = 1

# we assume m = 1 and r = 1
omega = 2 * np.pi


def phi(t):
    return omega * np.sin(t)


def phi_p(t):
    return omega * np.cos(t)


# force = phi_pp
def phi_pp(t):
    return -omega * np.sin(t)


match INDEX:
    case 0:

        def F(t, vy, vyp):
            x, y, u, v = vy
            x_dot, y_dot, u_dot, v_dot = vyp

            force = phi_pp(t)

            R = np.zeros(4, dtype=np.common_type(vy, vyp))

            la = -0.5 * (u**2 + v**2) / (x**2 + y**2)
            R[0] = x_dot - u
            R[1] = y_dot - v
            R[2] = u_dot - (2 * x * la - y * force)
            R[3] = v_dot - (2 * y * la + x * force)

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                ]
            )

            return y, yp

    case _:

        def F(t, vy, vyp):
            x, y, u, v, la = vy
            x_dot, y_dot, u_dot, v_dot, _ = vyp

            force = phi_pp(t)

            R = np.zeros(5, dtype=np.common_type(vy, vyp))

            match INDEX:
                case 1:
                    R[4] = 2 * (x * u_dot + u**2 + y * v_dot + v**2)

                case 2:
                    R[4] = 2 * (x * u + y * v)

                case 3:
                    R[4] = x**2 + y**2 - 1

            R[0] = x_dot - u
            R[1] = y_dot - v
            R[2] = u_dot - (2 * x * la - y * force)
            R[3] = v_dot - (2 * y * la + x * force)

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -phi_p(t) ** 2 / 2,
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                    -phi_pp(t),
                ]
            )

            return y, yp


problem = DAEProblem(
    name="circular_motion",
    F=F,
    t_span=(0.1 * np.pi, 0.45 * np.pi),
    index=INDEX,
    true_sol=true_sol,
)


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    # h = 5e-2
    h = 1e-2
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        # sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y

    # visualization
    n = len(y0)
    fig, ax = plt.subplots(n, 1)

    y_true, yp_true = true_sol(t)

    for i in range(n):
        ax[i].plot(t, y[:, i], "-r", label=f"y{i}")
        ax[i].plot(t, y_true[i], "x", label=f"y{i} true")
        ax[i].grid()
        ax[i].legend()

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    pow_min = 0
    # pow_max = 6
    # pow_max = 8
    pow_max = 10
    h_max = Dt / 4
    h0s = h_max * (1 / 2) ** (np.arange(pow_min, pow_max, dtype=float))

    rtols = 1e-16 * np.ones_like(h0s)
    atols = 1e-16 * np.ones_like(h0s)

    print(f"rtols: {rtols}")
    print(f"atols: {atols}")
    print(f"h0s: {h0s}")

    errors, rates = convergence_analysis(
        problem,
        rtols,
        atols,
        h0s,
    )


if __name__ == "__main__":
    trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)

    # convergence()
