import numpy as np
from scipy.linalg import qr


def householder_qr_pivoting(A):
    """QR-decomposition with column pivoting.

    See https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    and https://www.netlib.org/lapack/lawnspdf/lawn114.pdf.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    p = np.arange(n)

    for k in range(min(m, n)):
        # compute norms of remaining columns for pivoting
        norms = np.linalg.norm(R[k:, k:], axis=0)
        max_col = np.argmax(norms) + k

        # swap columns in R and update p and col_norms accordingly
        R[:, [k, max_col]] = R[:, [max_col, k]]
        p[[k, max_col]] = p[[max_col, k]]

        # Householder reflection for k-th column
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        sign = -np.sign(x[0]) if x[0] != 0 else -1.0
        u1 = x[0] - sign * norm_x
        w = x / u1
        w[0] = 1.0
        tau = -sign * u1 / norm_x

        # apply reflection to R
        R[k:, :] -= np.outer(tau * w, w @ R[k:, :])

        # apply reflection to Q
        Q[:, k:] -= np.outer(Q[:, k:] @ w, tau * w)

    return Q, R, p


# fmt: off
eps = 1e-6
A = np.array([
    [1,     1],
    [0, eps],
])
den2 = 1 + eps**2
den = np.sqrt(den2)
Q_true = np.array([
    [  1 / den, eps / den2],
    [eps / den,  -1 / den2],
])
R_true = np.array([
    [den,    1 / den],
    [  0, eps / den2],
])
p_true = np.array([1, 0], dtype=int)
# fmt: on

Q, R, p = qr(A, pivoting=True)
Q_, R_, p_ = householder_qr_pivoting(A)

Q = Q_
R = R_
p = p_

P = np.eye(len(p))[:, p]
print(f"Q:\n{Q}")
print(f"R:\n{R}")
print(f"p: {p}")
print(f"P:\n{P}")

assert np.allclose(A[:, p], Q @ R)
assert np.allclose(A @ P, Q @ R)
assert np.allclose(A, Q @ R @ P.T)
