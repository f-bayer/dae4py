import numpy as np
from dae4py.dae_problem import DAEProblem


# Problem parameters
Ub = 6
R0 = 1000
Ri = 9000
R1, R2, R3, R4, R5 = 9000 * np.ones(5)
alpha = 0.99
beta = 1e-6
Uf = 0.026
Ue = lambda t: 0.4 * np.sin(200 * np.pi * t)
f = lambda U: beta * (np.exp(U / Uf) - 1)
C1, C2, C3 = 1e-6 * np.arange(1, 4)

# initial states
y0 = np.zeros(5)
y0[1] = Ub * R1 / (R1 + R2)
y0[2] = Ub * R1 / (R1 + R2)
y0[3] = Ub

# initial derivatives
yp0 = np.zeros_like(y0)
yp0[2] = (f(y0[1] - y0[2]) - y0[2] / R3) / C2


def F(t, y, yp):
    U1, U2, U3, U4, U5 = y
    Up1, Up2, Up3, Up4, Up5 = yp

    f23 = f(U2 - U3)

    return np.array(
        [
            (Ue(t) - U1) / R0 + C1 * (Up2 - Up1),
            (Ub - U2) / R2 - U2 / R1 + C1 * (Up1 - Up2) - (1 - alpha) * f23,
            f23 - U3 / R3 - C2 * Up3,
            (Ub - U4) / R4 + C3 * (Up5 - Up4) - alpha * f23,
            -U5 / R5 + C3 * (Up4 - Up5),
        ]
    )


problem = DAEProblem(
    name="Transistor amplifier",
    F=F,
    t_span=(0, 0.2),
    index=1,
    y0=y0,
    yp0=yp0,
    parameters={"Ue": Ue},
)
