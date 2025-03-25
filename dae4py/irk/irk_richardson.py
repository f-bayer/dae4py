import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy._lib._util import _RichResult

from dae4py.math import newton, simplified_newton

from scipy.optimize import root, fsolve


class RungeKuttaBase(ABC):
    def __init__(
        self, f, y0, yp0, t_span, h0, tableau, atol, rtol, newton_max_iter=None
    ):
        self.f = f

        self.t0, self.t1 = t_span
        if self.t1 <= self.t0:
            raise ValueError("t1 must be greater than t0")

        # ensure that initial conditions are numpy arrays
        self.y0, self.yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
        self.m = len(self.y0)

        self.tableau = tableau

        self.h0 = h0
        self.atol, self.rtol = atol, rtol
        if newton_max_iter is None:
            newton_max_iter = 7 + int((tableau.s - 3) * 2.5)

        self.newton_max_iter = newton_max_iter

    def residual(self, tn, yn, hn):
        # extract necessary parts of the Butcher talbuea
        A, c, s = self.tableau.A, self.tableau.c, self.tableau.s

        # precompute stage times
        T = tn + c * hn

        def fun(Yp_flat):
            # reshape flat input to stage derivatives
            Yp = Yp_flat.reshape(s, -1)

            # compute stage values
            Y = yn + hn * A.dot(Yp)

            # residuals for all stages
            F = np.zeros_like(Yp)
            for i in range(s):
                F[i] = self.f(T[i], Y[i], Yp[i])
            return F.flatten()

        return fun

    @abstractmethod
    def solve_nonlinear_system(
        self, fun, x, atol, rtol, newton_max_iter, *args, **kwargs
    ): ...

    @abstractmethod
    def estimate_error(self): ...

    @abstractmethod
    def predict_factor(self): ...

    def step(self, tn, yn, hn, Yp, *args, **kwargs):
        fun = self.residual(tn, yn, hn)
        sol = self.solve_nonlinear_system(
            fun,
            Yp.reshape(-1),
            self.atol,
            self.rtol,
            self.newton_max_iter,
            *args,
            **kwargs,
        )
        # if not sol.success:
        #     raise RuntimeError(
        #         f"Newton solver failed at t={t0 + hn} with error={sol.error:.2e}"
        #     )

        # extract the solution for stages
        Yp = sol.x.reshape(self.tableau.s, -1)
        Y = yn + hn * self.tableau.A.dot(Yp)

        # update y and y'
        y1 = yn + hn * self.tableau.b.dot(Yp)
        # # TODO: Use correct formula from the lecture notes
        # b = self.tableau.b
        # s = self.tableau.s
        # W = np.linalg.inv(self.tableau.A)
        # y1 = (1 - b.T @ W @ np.ones(s)) * yn + b.T @ W @ Y
        yp1 = Yp[-1]  # only correct for stiffly accurate methods

        return _RichResult(
            y1=y1, yp1=yp1, Y=Y, Yp=Yp, nit=sol.nit, rate=sol.rate, newton_sol=sol
        )

    def solve(self):
        A = self.tableau.A
        s = self.tableau.s

        # initial guess for stage derivatives
        Yp = np.tile(self.yp0, s).reshape(s, -1)
        Y = self.y0 + self.h0 * A.dot(Yp)

        # initialize solution arrays
        h_sol = [self.h0]
        t_sol = [self.t0]
        y_sol = [self.y0]
        yp_sol = [self.yp0]
        Y_sol = [Y]
        Yp_sol = [Yp]

        tn = self.t0
        hn = self.h0
        yn = self.y0

        # build progress bar
        offset = min(self.t0, self.t1)
        frac = (self.t1 - self.t0) / 100
        pbar = tqdm(total=100, leave=True)
        i = 0

        error_norm_old = None
        hn_old = None
        while tn < self.t1:
            step_accepted = False
            while not step_accepted:
                sol = self.step(tn, yn, hn, Yp)
                error_norm = self.estimate_error(tn, hn, yn, Yp, sol.y1, sol)
                factor = self.predict_factor(
                    hn, hn_old, error_norm, error_norm_old, sol.newton_sol
                )

                if error_norm < 1:
                    step_accepted = True
                    break
                else:
                    # only update step-size if step is not accepted
                    hn *= factor

            t_sol.append(tn + hn)
            h_sol.append(hn)
            y_sol.append(sol.y1.copy())
            yp_sol.append(sol.yp1.copy())
            Y_sol.append(sol.Y.copy())
            Yp_sol.append(sol.Yp.copy())

            yn = sol.y1.copy()
            tn = tn + hn

            # update progress bar
            i1 = int((tn - offset) // frac)
            pbar.update(i1 - i)
            pbar.set_description(f"t: {tn:0.2e}s < {self.t1:0.2e}s; h: {hn:0.2e}")
            i = i1

            # store old error norm and step-size
            hn_old = hn
            error_norm_old = error_norm

            # update next step-size
            hn *= factor

        return _RichResult(
            t=np.array(t_sol),
            h=np.array(h_sol),
            y=np.array(y_sol),
            yp=np.array(yp_sol),
            Y=np.array(Y_sol),
            Yp=np.array(Yp_sol),
        )


class SimpleRungeKutta(RungeKuttaBase):

    def solve_nonlinear_system(
        self, fun, x, atol, rtol, newton_max_iter, *args, **kwargs
    ):
        sol = root(fun, x, tol=atol, method="lm")
        sol.rate = None
        sol.nit = sol.nfev
        return sol
        # print(f"")
        # return newton(fun, x, atol=atol, rtol=rtol, max_iter=newton_max_iter)
        # return simplified_newton(fun, x, atol=atol, rtol=rtol, max_iter=newton_max_iter)

    def estimate_error(self, *args):
        return 0.0

    def predict_factor(self, *args):
        return 1.0


class AdaptiveRungeKuttaRichardson(RungeKuttaBase):

    def solve_nonlinear_system(
        self, fun, x, atol, rtol, newton_max_iter, *args, LU=None, **kwargs
    ):
        return newton(fun, x, atol=atol, rtol=rtol, max_iter=newton_max_iter)
        # return simplified_newton(
        #     fun, x, atol=atol, rtol=rtol, max_iter=newton_max_iter, LU=LU
        # )

    def estimate_error(self, tn, hn, yn, Yp, w1, sol):
        # reuse LU decomposition
        # LU = sol.newton_sol.LU
        LU = None

        # compute two consecutive half-steps
        sol_half = self.step(tn, yn, hn / 2, Yp, LU=LU)
        sol1 = self.step(tn + hn / 2, sol_half.y1, hn / 2, sol_half.Yp, LU=LU)

        # richardson extrapolation (error estimate)
        error = (sol1.y1 - w1) / (2**self.tableau.p - 1)
        scale = self.atol + np.maximum(np.abs(yn), np.abs(sol1.y1)) * self.rtol
        error_norm = np.linalg.norm(error / scale) / error.size**0.5
        # print(f"error_norm: {error_norm}")
        return error_norm

    def predict_factor(self, h, h_old, error_norm, error_norm_old, newton_sol):
        rate = newton_sol.rate
        nit = newton_sol.nit
        if rate > 1:
            # print(f"Newton rate > 1; half step-size")
            return 0.5
        else:
            # adapt step-size
            if h_old is not None and error_norm_old is not None:
                # if False:
                # predictive controller
                with np.errstate(invalid="ignore", divide="ignore"):
                    multiplier = (h / h_old) * (error_norm_old / error_norm) ** (
                        1 / (self.tableau.p + 1)
                    )
            else:
                multiplier = 1.0

            # use factor that leads to a smaller step-size
            with np.errstate(divide="ignore"):
                factor = min(1, multiplier) * error_norm ** (-1 / (self.tableau.p + 1))

            # smooth limiter
            # KAPPA = 0.75
            KAPPA = 0.5
            # KAPPA = 0.25
            # KAPPA = 0.125
            factor = 1 + KAPPA * np.arctan((factor - 1) / KAPPA)

            # savety factor depending on required newton iterations
            safety = (
                0.9 * (2 * self.newton_max_iter + 1) / (2 * self.newton_max_iter + nit)
            )
            return safety * factor


if False:

    def solve_dae_IRK_generic(f, y0, yp0, t_span, h0, tableau, atol=1e-6, rtol=1e-6):
        """
        Solves a system of DAEs using an implicit Runge-Kutta method.

        Parameters:
            f: Function defining the DAE system, f(t, y, yp) = 0.
            y0: Initial condition for y.
            yp0: Initial condition for y'.
            t_span: Tuple (t0, t1) defining the time span.
            h: Step size.
            tableau: Butcher tableau coefficients for the IRK method.
            atol: Absolute tolerance for the Newton solver.
            rtol: Relative tolerance for the Newton solver.

        Returns:
            _RichResult containing time points, solutions y, and derivatives yp.
        """
        t0, t1 = t_span
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")

        A, b, c, p, q, s = (
            tableau.A,
            tableau.b,
            tableau.c,
            tableau.p,
            tableau.q,
            tableau.s,
        )

        y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
        m = len(y0)
        s = len(c)
        newton_max_iter = 7 + int((s - 3) * 2.5)

        # initial guess for stage derivatives
        Yp = np.tile(yp0, s).reshape(s, -1)
        Y = y0 + h0 * A.dot(Yp)

        def step(tn, yn, hn, Yp):
            # precompute stage times
            T = tn + c * hn

            def residual(Yp_flat):
                # reshape flat input to stage derivatives
                Yp = Yp_flat.reshape(s, -1)

                # compute stage values
                Y = yn + hn * A.dot(Yp)

                # residuals for all stages
                F = np.zeros((s, m))
                for i in range(s):
                    F[i] = f(T[i], Y[i], Yp[i])
                return F.flatten()

            # solve the nonlinear system
            # sol = newton(residual, Yp.flatten(), atol=atol, rtol=rtol)
            sol = simplified_newton(residual, Yp.flatten(), atol=atol, rtol=rtol)
            print(f"sol.nit: {sol.nit}")
            if not sol.success:
                raise RuntimeError(
                    f"Newton solver failed at t={t0 + hn} with error={sol.error:.2e}"
                )

            # extract the solution for stages
            Yp = sol.x.reshape(s, -1)
            Y = yn + hn * A.dot(Yp)

            # update y and y'
            y1 = yn + hn * b.dot(Yp)
            yp1 = Yp[-1]  # only correct for stiffly accurate methods

            return y1, yp1, Y, Yp, sol.nit

        # initialize solution arrays
        h = [h0]
        t = [t0]
        y = [y0]
        yp = [yp0]
        Ys = [Y]
        Yps = [Yp]

        steps = int(np.ceil((t1 - t0) / h0))
        with tqdm(total=steps, desc="IRK Integration") as pbar:
            while t0 < t1:
                step_accepted = False
                h1 = h0
                while not step_accepted:
                    # compute two consecutive half-steps
                    y1_half, yp1_half, Y_half, Yp_half, nit_half = step(
                        t0, y0, h1 / 2, Yp
                    )
                    y1, yp1, Y, Yp, nit = step(t0 + h1 / 2, y1_half, h1 / 2, Yp_half)

                    # compute the full step
                    w1, wp1, W, Wp, nit_w = step(t0, y0, h1, Yp)

                    # richardson extrapolation (error estimate)
                    error = (y1 - w1) / (2**p - 1)
                    scale = atol + np.maximum(np.abs(y), np.abs(y1)) * rtol
                    error_norm = np.linalg.norm(error / scale) / error.size**0.5
                    print(f"error_norm: {error_norm}")

                    # adapt step-size
                    factor = error_norm ** (-1 / (p + 1))
                    # smooth limiter
                    # KAPPA = 0.5
                    # factor = 1 + KAPPA * np.arctan((factor - 1) / KAPPA)
                    safety = (
                        0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + nit_w)
                    )
                    h1 *= safety * factor
                    print(f"h: {h0}")
                    if error_norm < 1:
                        step_accepted = True

                # append to solution arrays
                h.append(h0)
                t.append(t0 + h0)
                y.append(y1)
                yp.append(yp1)
                Ys.append(Y)
                Yps.append(Yp)

                # advance time and update initial values
                t0 += h0
                y0 = y1

                # choose new step-size for next step
                h0 = h1

                pbar.update(1)

        return _RichResult(
            h=np.array(h),
            t=np.array(t),
            y=np.array(y),
            yp=np.array(yp),
            Y=np.array(Ys),
            Yp=np.array(Yps),
        )


if True:

    def solve_dae_IRK_generic(
        f,
        y0,
        yp0,
        t_span,
        h0,
        tableau,
        Method=SimpleRungeKutta,
        # Method=AdaptiveRungeKuttaRichardson,
        atol=1e-4,
        rtol=1e-4,
        newton_max_iter=None,
        *args,
        **kwargs,
    ):
        method = Method(
            f,
            y0,
            yp0,
            t_span,
            h0,
            tableau,
            atol,
            rtol,
            newton_max_iter,
            *args,
            **kwargs,
        )
        return method.solve()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau

    t0 = 0
    t1 = 3
    t_span = (t0, t1)

    la = 3

    def f(t, y, yp):
        return yp + la * y

    def y_true(t):
        return np.exp(-la * (t - t0))

    y0 = y_true(t0)
    yp0 = -la * y0

    s = 2
    tableau = radau_tableau(s)
    # tableau = gauss_tableau(s)

    h = 1e-1

    sol = solve_dae_IRK_generic(f, y0, yp0, t_span, h, tableau)
    t = sol.t
    y = sol.y
    yp = sol.yp

    t_dense = np.linspace(t[0], t[-1])

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(t_dense, y_true(t_dense), "-k", label="y(t) true")
    ax.plot(t_dense, -la * y_true(t_dense), "-b", label="y'(t) true")
    ax.plot(t, y, "--or", label="y(t)")
    ax.plot(t, yp, "--xg", label="y'(t)")
    ax.grid()
    ax.legend()
    plt.show()
