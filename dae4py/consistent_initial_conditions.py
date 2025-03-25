import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.integrate._ivp.common import norm, EPS
from scipy.optimize._numdiff import approx_derivative


def consistent_initial_conditions(
    F,
    t0,
    y0,
    yp0,
    jac=None,
    fixed_y0=None,
    fixed_yp0=None,
    rtol=1e-6,
    atol=1e-6,
    newton_maxiter=10,
    chord_iter=3,
    safety=0.5,
    *args,
):
    """Compute consistent initial conditions for DAE problem."""
    n = len(y0)

    if jac is None:

        def jac(t, y, yp):
            n = len(y)
            z = np.concatenate((y, yp))

            def fun_composite(t, z):
                y, yp = z[:n], z[n:]
                return F(t, y, yp)

            J = approx_derivative(lambda z: fun_composite(t, z), z, method="2-point")
            J = J.reshape((n, 2 * n))
            Jy, Jyp = J[:, :n], J[:, n:]
            return Jy, Jyp

    if fixed_y0 is None:
        free_y = np.arange(n)
    else:
        free_y = np.setdiff1d(np.arange(n), fixed_y0)

    if fixed_yp0 is None:
        free_yp = np.arange(n)
    else:
        free_yp = np.setdiff1d(np.arange(n), fixed_yp0)

    if len(free_y) + len(free_yp) < n:
        raise ValueError(f"Too many components fixed, cannot solve the problem.")

    if not (isinstance(rtol, float) and rtol > 0):
        raise ValueError("Relative tolerance must be a positive scalar.")

    if rtol < 100 * EPS:
        rtol = 100 * EPS
        print(f"Relative tolerance increased to {rtol}")

    if np.any(np.array(atol) <= 0):
        raise ValueError("Absolute tolerance must be positive.")

    assert 0 < safety <= 1, "safety factor has to be in (0, 1]"

    y0 = np.asarray(y0, dtype=float).reshape(-1)
    yp0 = np.asarray(yp0, dtype=float).reshape(-1)
    f = F(t0, y0, yp0, *args)
    Jy, Jyp = jac(t0, y0, yp0)

    scale_f = atol + np.abs(f) * rtol
    for _ in range(newton_maxiter):
        for _ in range(chord_iter):
            dy, dyp = solve_underdetermined_system(f, Jy, Jyp, free_y, free_yp)
            y0 += dy
            yp0 += dyp

            f = F(t0, y0, yp0, *args)
            error = norm(f / scale_f)
            if error < safety:
                return y0, yp0, f

        Jy, Jyp = jac(t0, y0, yp0)

    raise RuntimeError("Convergence failed.")


def qr_rank(A):
    """Compute QR-decomposition with column pivoting of A and estimate the rank."""
    Q, R, p = qr(A, pivoting=True)
    tol = max(A.shape) * EPS * abs(R[0, 0])
    rank = np.sum(abs(np.diag(R)) > tol)
    return rank, Q, R, p


def solve_underdetermined_system(f, Jy, Jyp, free_y, free_yp):
    """Solve the underdetermined system
        0 = f + Jy @ Delta_y + Jyp @ Delta_yp
    A solution is obtained with as many components as possible of
    (transformed) Delta_y and Delta_yp set to zero.
    """
    n = len(f)
    Delta_y = np.zeros(n)
    Delta_yp = np.zeros(n)

    # handel special cases first
    fixed = (n - len(free_y)) + (n - len(free_yp))

    if len(free_y) == 0:
        # solve 0 = f + Jyp @ Delta_yp (ODE case)
        rank, Q, R, p = qr_rank(Jyp)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(
                    f"Too many fixed components, rank deficiency is {rankdef}."
                )
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        Delta_yp_ = np.zeros_like(Delta_yp)
        Delta_yp_[p] = solve_triangular(R, d)
        Delta_yp[free_yp] = Delta_yp_
        return Delta_y, Delta_yp

    if len(free_yp) == 0:
        # solve 0 = f + Jy @ Delta_y (pure algebraic case)
        rank, Q, R, p = qr_rank(Jy)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(
                    f"Too many fixed components, rank deficiency is {rankdef}."
                )
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        Delta_y_ = np.zeros_like(Delta_y)
        Delta_y_[p] = solve_triangular(R, d)
        Delta_y[free_y] = Delta_y_
        return Delta_y, Delta_yp

    # eliminate variables that are not free
    Jy = Jy[:, free_y]
    Jyp = Jyp[:, free_yp]

    # QR-decomposition of Fyp leads to the system
    # [R11, R12] [w1'] + [S11, S12] [w1] = [d1]
    # [  0,   0] [w2'] + [S21, S22] [w2] = [d2]
    # with S = Q.T @ Fy
    rank, Q, R, p = qr_rank(Jyp)
    d = -Q.T @ f
    if rank == n:
        # Full rank (ODE) case:
        # Set all free dy to zero and solve triangular system below
        Delta_y[free_y] = 0
        Delta_yp_ = np.zeros_like(Delta_yp)
        Delta_yp_[p] = solve_triangular(R, d)
        Delta_yp[free_yp] = Delta_yp_
    else:
        # Rank deficient (DAE) case:
        S = Q.T @ Jy
        rankS, QS, RS, pS = qr_rank(S[rank:])
        rankdef = n - (rank + rankS)
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(
                    f"Too many fixed components, rank deficiency is {rankdef}."
                )
            else:
                raise ValueError("Index greater than one.")

        # decompose d
        d1 = d[:rank]
        d2 = d[rank:]

        # compute basic solution of underdetermined system
        # [S21, S22] [w1] = d2
        #            [w2]
        # using column pivoting QR-decomposition
        # [RS11, RS12] [v1] = [c1]
        # [   0,    0] [v2]   [c2]
        # with v2 = 0 this gives
        # RS11 @ v1 = c1
        c = QS.T @ d2
        v = np.zeros(RS.shape[1])
        v[:rankS] = solve_triangular(RS[:rankS, :rankS], c[:rankS])

        # apply permutation
        w = np.zeros_like(v)
        w[pS] = v

        # set w2' = 0 and solve the remaining system
        # [R11] w1' = d1 - [S11, S12] [w1]
        #                             [w2]
        wp = np.zeros(R.shape[1])
        if rank > 0:
            wp_ = np.zeros(R.shape[1])
            wp_[:rank] = solve_triangular(R[:rank, :rank], d1 - S[:rank] @ w)
            wp[p] = wp_

        # store w and wp for free dy and dyp
        Delta_y[free_y] = w
        Delta_yp[free_yp] = wp

    return Delta_y, Delta_yp
