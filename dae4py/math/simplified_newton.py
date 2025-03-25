import numpy as np
from scipy._lib._util import _RichResult
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate._ivp.common import EPS


def simplified_newton(
    fun,
    x0,
    jac="2-point",
    atol=1e-6,
    rtol=1e-6,
    max_iter=20,
    LU=None,
):
    nfev = 0
    njev = 0

    # wrap function
    def fun(x, f=fun):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(f(x))

    # wrap jacobian or use a finite difference approximation
    if callable(jac):

        def jacobian(x):
            nonlocal njev
            njev += 1
            return jac(x)

    elif jac in ["2-point", "3-point", "cs"]:

        def jacobian(x):
            nonlocal njev
            njev += 1
            return approx_derivative(
                lambda y: fun(y),
                x,
                method=jac,
            )

    else:
        raise RuntimeError

    # newton tolerance
    # newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
    # newton_tol = min(0.03, rtol ** 0.5)
    newton_tol = min(0.03, rtol**0.5)
    # print(f"newton_tol: {newton_tol}")

    # eliminate round-off errors
    Delta_x = np.zeros_like(x0)
    x = x0 + Delta_x

    # evaluate Jacobian at initial point
    if LU is None:
        J = np.atleast_2d(jacobian(x))
        LU = lu_factor(J)

    # scaling with relative and absolute tolerances
    scale = atol + np.abs(x) * rtol

    # Newton loop
    # error = None
    error = np.inf
    dx_norm_old = None
    rate = None
    i = 0
    converged = False
    if not converged:
        for i in range(max_iter):
            # new function value
            f = np.atleast_1d(fun(x))

            # Newton update
            # dx = np.linalg.solve(J, f)
            dx = lu_solve(LU, f)

            # perform Newton step
            Delta_x -= dx
            x = x0 + Delta_x

            # check convergence rate
            dx_norm = np.linalg.norm(dx / scale) / scale.size**0.5
            if dx_norm_old is not None:
                rate = dx_norm / dx_norm_old

            # check for divergence
            if rate is not None and (
                rate >= 1 or rate ** (max_iter - i) / (1 - rate) * dx_norm > newton_tol
            ):
                break

            # check for convergence
            if (
                dx_norm == 0
                or rate is not None
                and rate / (1 - rate) * dx_norm < newton_tol
            ):
                converged = True
                break

            dx_norm_old = dx_norm

        # if not converged:
        #     raise RuntimeError(
        #         f"simplified_newton is not converged after {i + 1} iterations with error {error:.2e}"
        #     )

    return _RichResult(
        x=x,
        success=converged,
        error=error,
        fun=f,
        nit=i + 1,
        nfev=nfev,
        njev=njev,
        rate=rate,
        LU=LU,
    )
