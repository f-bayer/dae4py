import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""Modified index 2 DAE found in Jay1993 Example 7.

References:
-----------
Jay1993: https://link.springer.com/article/10.1007/BF01990349
"""


def F(t, y, yp, nonlinear_multiplier=False):
    y1, y2, _ = y
    y1p, y2p, la = yp

    F = np.zeros(3, dtype=y.dtype)
    if nonlinear_multiplier:
        F[0] = y1p - (y1 * y2**2 * la**2)
    else:
        F[0] = y1p - (y1 * y2 * la)
    F[1] = y2p - (y1**2 * y2**2 - 3 * y2**2 * la)
    F[2] = y1**2 * y2 - 1.0

    return F


def sol_true(t):
    y = np.array(
        [
            np.exp(t),
            np.exp(-2 * t),
            0.5 * np.exp(2 * t),
        ]
    )

    yp = np.array(
        [
            np.exp(t),
            -2 * np.exp(-2 * t),
            np.exp(2 * t),
        ]
    )

    return y, yp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-6

    # initial conditions
    y0, yp0 = sol_true(t0)

    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    start = time.time()
    # method = "BDF"
    method = "Radau"
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    tp = t
    yp = sol.yp
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # errors
    y_true, yp_true = sol_true(t)
    error_y1 = np.linalg.norm(y[0] - y_true[0]) / np.linalg.norm(y_true[0])
    error_y2 = np.linalg.norm(y[1] - y_true[1]) / np.linalg.norm(y_true[1])
    error_la = np.linalg.norm(y[2] - y_true[2]) / np.linalg.norm(y_true[2])
    print(f"error y: [{error_y1}, {error_y2}, {error_la}]")

    error_y1p = np.linalg.norm(yp[0] - yp_true[0]) / np.linalg.norm(yp_true[0])
    error_y2p = np.linalg.norm(yp[1] - yp_true[1]) / np.linalg.norm(yp_true[1])
    error_lap = np.linalg.norm(yp[2] - yp_true[2]) / np.linalg.norm(yp_true[2])
    print(f"error yp: [{error_y1p}, {error_y2p}, {error_lap}]")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, y[1], "--g", label="y2")
    ax.plot(tp, yp[2], "--b", label="la")

    ax.plot(t, y_true[0], "-r", label="y1 true")
    ax.plot(t, y_true[1], "-g", label="y2 true")
    ax.plot(t, yp_true[2], "-b", label="la true")

    ax.grid()
    ax.legend()

    plt.show()
