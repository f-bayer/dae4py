import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py import dassl, pside, radau

omega = 3
eps = 1e3
la = -20

phi1 = lambda t: np.arctan(eps * np.cos(omega * t))
phi2 = lambda t: np.sin(t)

phi1_dot = lambda t: -eps * omega * np.sin(omega * t) / (1 + (eps * np.cos(omega * t))**2)
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

if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)
    # t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None

    # initial conditions
    y0, yp0 = true_sol(t0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")

    # tolerances
    # rtol = atol = 1e-6
    rtol = atol = 1e-10

    start = time.time()
    sol = dassl(F, t_span, y0, yp0, rtol=rtol, atol=atol, t_eval=t_eval)
    # sol = pside(F, t_span, y0, yp0, rtol=rtol, atol=atol)
    # sol = radau(F, t_span, y0, yp0, rtol=rtol, atol=atol)
    end = time.time()
    print(f"elapsed time: {end - start}")
    # print(sol)

    success = sol["success"]
    t = sol["t"]
    y = sol["y"]
    yp = sol["yp"]
    assert success

    print(f"t.shape: {t.shape}")
    print(f"y.shape: {y.shape}")
    print(f"yp.shape: {yp.shape}")

    # error
    y_true, yp_true = true_sol(t)
    error_y = np.max(np.linalg.norm(y.T - y_true, axis=0))
    error_yp = np.max(np.linalg.norm(yp.T - yp_true, axis=0))
    print(f"error y : {error_y}")
    print(f"error yp: {error_yp}")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[:, 0], "--or", label="y1")
    ax.plot(t, y[:, 1], "--og", label="y2")

    ax.plot(t, y_true[0], "-r", label="y1 true")
    ax.plot(t, y_true[1], "-g", label="y2 true")

    ax.grid()
    ax.legend()

    plt.show()