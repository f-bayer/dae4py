import time
import numpy as np
import matplotlib.pyplot as plt
from psidemodule import integrate


def F(t, y, yp):
    # print(f"F(t, y, yp) called with:")
    # print(f" - t: {t}")
    # print(f" - y: {y}")
    # print(f" - yp: {yp}")
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - t * y2p + y1 - (1 + t) * y2
    F[1] = y2 - np.sin(t)

    return F

if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e2
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    # t_eval = None

    # tolerances
    rtol = atol = 9.8e-11

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)

    start = time.time()
    sol = integrate(F, t_span, y0, yp0, rtol=rtol, atol=atol, t_eval=t_eval)
    end = time.time()
    print(f"elapsed time: {end - start}")

    success = sol["success"]
    t = sol["t"]
    y = sol["y"]
    yp = sol["yp"]
    assert success

    # error
    diff = y[-1] - np.array([
        np.exp(-t1) + t1 * np.sin(t1),
        np.sin(t1),
    ])
    error = np.linalg.norm(diff)
    print(f"error: {error}")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[:, 0], "--r", label="y1")
    ax.plot(t, y[:, 1], "--g", label="y2")

    ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    ax.plot(t, np.sin(t), "-g", label="y2 true")

    ax.grid()
    ax.legend()

    plt.show()