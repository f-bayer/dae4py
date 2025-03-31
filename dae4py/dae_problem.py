import numpy as np


class DAEProblem:
    def __init__(
        self,
        name,
        F,
        t_span,
        index,
        y0=None,
        yp0=None,
        true_sol=None,
        jac=None,
        parameters=None,
    ):
        """
        Class representing a Differential-Algebraic Equation (DAE) problem.

        Parameters
        ----------
        name: str
            The name of the DAE problem.
        F: callable
            The function representing the DAE system with signature F(t, y, y') = 0.
        t_span: tuple
            The time span for the problem, represented as a tuple (t0, t1).
        t0: float
            The initial time.
        t1: float
            The final time.
        index: int
            The index of the DAE system.
        y0: array-like
            The initial values of the unknown variables `y` at time `t0`.
        yp0: array-like
            The initial values of the derivatives of the unknown variables `y`
            at time `t0`.
        true_sol: callable
            A function that returns the true solution `(y, yp)` of the DAE
            system at a given time `t`.
        jac: callable, optional, default: None
            A function that returns the Jacobian of the system with respect to
            `y` and `yp`.
        parameters: dict, optional, default: {}
            A dictionary of additional parameters required for the system.
        """
        self.name = name
        self.F = F
        self.t_span = t_span
        self.t0, self.t1 = t_span
        self.index = index
        if y0 is None or yp0 is None:
            if true_sol is not None:
                self.y0, self.yp0 = true_sol(self.t0)
            else:
                raise ValueError(
                    "Either `y0` and `yp0` must be provided, or `true_sol` must be given."
                )
        else:
            self.y0 = np.array(y0)
            self.yp0 = np.array(yp0)
        self.true_sol = true_sol
        self.jac = jac
        self.parameters = parameters if parameters else {}
