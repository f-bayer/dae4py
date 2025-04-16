import numpy as np
from dae4py.consistent_initial_conditions import consistent_initial_conditions


def fun(t, y, yp):
    return np.array(
        [
            2 * yp[0] - y[1],
            y[0] + y[1],
        ]
    )


if __name__ == "__main__":
    t0 = 0
    y0 = [1, 0]
    yp0 = [0, -3]

    F0 = fun(t0, y0, yp0)
    assert not np.allclose(F0, np.zeros_like(F0))

    # # alter all initial values => flawed initial conditions
    # fixed_y0 = []
    # fixed_yp0 = []
    # keep y1 = 1 => correct initial conditions
    fixed_y0 = [0]
    fixed_yp0 = []

    y0, yp0, F0 = consistent_initial_conditions(
        fun,
        t0,
        y0,
        yp0,
        fixed_y0=fixed_y0,
        fixed_yp0=fixed_yp0,
    )
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    assert np.allclose(F0, np.zeros_like(F0))
