import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py import dassl, pside, radau


def F(t, y, yp):
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
    # t_eval = np.linspace(t0, t1, num=100)
    t_eval = None

    # tolerances
    # rtol = atol = 9.8e-11
    # rtol = atol = 1e-12
    rtol = atol = 1e-6

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)

    print(dassl.__doc__)

    exit()

    start = time.time()
    # sol = dassl(F, t_span, y0, yp0, rtol=rtol, atol=atol)
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
    
    print(f"t: {t}")
    # exit()

    # error
    diff = y[-1] - np.array([
        np.exp(-t1) + t1 * np.sin(t1),
        np.sin(t1),
    ])
    error = np.linalg.norm(diff)
    print(f"error: {error}")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[:, 0], "--or", label="y1")
    ax.plot(t, y[:, 1], "--og", label="y2")

    ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    ax.plot(t, np.sin(t), "-g", label="y2 true")

    ax.grid()
    ax.legend()

    plt.show()