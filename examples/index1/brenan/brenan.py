import numpy as np
from dae4py.dae_problem import DAEProblem


def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - t * y2p + y1 - (1 + t) * y2
    F[1] = y2 - np.sin(t)

    return F


def true_sol(t):
    return (
        np.array(
            [
                np.exp(-t) + t * np.sin(t),
                np.sin(t),
            ]
        ),
        np.array(
            [
                -np.exp(-t) + np.sin(t) + t * np.cos(t),
                np.cos(t),
            ]
        ),
    )


problem = DAEProblem(
    name="Brenan",
    F=F,
    t_span=(0, 20),
    index=1,
    true_sol=true_sol,
)
