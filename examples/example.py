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
    t1 = 1e3
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 9.8e-11

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)

    start = time.time()
    sol = integrate(F, t_span, y0, yp0, rtol=rtol, atol=atol)
    end = time.time()
    print(f"elapsed time: {end - start}")

    success = sol["success"]
    y = sol["y"]
    yp = sol["yp"]
    assert success

    # error
    diff = y - np.array([
        np.exp(-t1) + t1 * np.sin(t1),
        np.sin(t1),
    ])
    error = np.linalg.norm(diff)
    print(f"error: {error}")

    # print(f"y: {y}")
    # print(f"yp: {yp}")

    # t = sol.t
    # y = sol.y
    # tp = t
    # yp = sol.yp
    # success = sol.success
    # status = sol.status
    # message = sol.message
    # print(f"success: {success}")
    # print(f"status: {status}")
    # print(f"message: {message}")
    # print(f"nfev: {sol.nfev}")
    # print(f"njev: {sol.njev}")
    # print(f"nlu: {sol.nlu}")
    # print(f"y[:, -1]: {y[:, -1]}")

    # print(f"integrate(): {integrate()}")