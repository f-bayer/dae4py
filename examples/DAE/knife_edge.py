import numpy as np
import matplotlib.pyplot as plt
from dae4py.dae_problem import DAEProblem
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.benchmark import convergence_analysis


m = 1
Theta = 1
g = 9.81
Omega = 1
alpha = 35 / 180 * np.pi
salpha = np.sin(alpha)

"""Index 2 DAE found in Section 1.6 of Bloch2015.

References:
-----------
Bloch2015: https://doi.org/10.1007/978-1-4939-3017-3
"""


def F(t, vy, vyp):
    x, y, phi, u, v, omega, la = vy
    xp, yp, phip, up, vp, omegap, _ = vyp

    sphi, cphi = np.sin(phi), np.cos(phi)

    F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))

    # Bloch 2005, equation (1.7.6)
    F[0] = xp - u
    F[1] = yp - v
    F[2] = phip - omega
    F[3] = m * up + cphi * la
    F[4] = m * vp - sphi * la - m * g * salpha
    F[5] = Theta * omegap
    F[6] = v * sphi - u * cphi

    return F


def true_sol(t):
    x = (g * salpha / Omega) * (t / 2 - np.sin(2 * Omega * t) / (4 * Omega))
    y = (g * salpha / (2 * Omega**2)) * np.sin(Omega * t) ** 2
    phi = Omega * t

    u = (g * salpha / Omega) * np.sin(Omega * t) ** 2
    v = (g * salpha / Omega) * np.sin(Omega * t) * np.cos(Omega * t)
    omega = Omega * np.ones_like(t)

    la = -2 * m * g * salpha * np.sin(Omega * t)
    la_dot = -2 * m * g * salpha * Omega * np.cos(Omega * t)

    x_dot = u
    y_dot = v
    phi_dot = omega

    u_dot = -np.sin(Omega * t) * la / m
    v_dot = -g * salpha + np.cos(Omega * t) * la / m
    omega_dot = np.zeros_like(t)

    vy = np.array(
        [
            x,
            y,
            phi,
            u,
            v,
            omega,
            la,
        ]
    )

    vyp = np.array(
        [
            x_dot,
            y_dot,
            phi_dot,
            u_dot,
            v_dot,
            omega_dot,
            la_dot,
        ]
    )

    return vy, vyp


problem = DAEProblem(
    name="Extended Prothero-Robinson problem",
    F=F,
    t_span=(0.1 * np.pi, 0.5 * np.pi / Omega),
    index=1,
    true_sol=true_sol,
)


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-2
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
    fig, ax = plt.subplots(7, 1)

    y_true, yp_true = true_sol(t)

    for i in range(7):
        ax[i].plot(t, y[:, i], "-r", label=f"y{i}")
        ax[i].plot(t, y_true[i], "x", label=f"y{i} true")
        ax[i].grid()
        ax[i].legend()

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    # pow_min = 6
    # pow_max = 11
    pow_min = 4
    pow_max = 8
    h_max = Dt / 4
    h0s = h_max * (1 / 2) ** (np.arange(pow_min, pow_max, dtype=float))

    rtols = 1e-14 * np.ones_like(h0s)
    atols = 1e-14 * np.ones_like(h0s)

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
