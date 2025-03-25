import numpy as np
from scipy._lib._util import _RichResult


def radau_tableau(s):
    """
    Evaluate the Butcher tableau for the s-stage Radau IIA method.

    Parameters
    ----------
    s: int
        Number of stages.

    Returns
    -------
    butcher_tableau: _RichResult
        Container that stores
            - A (array-like): Coefficient matrix.
            - b (array-like): Quadrature weights.
            - c (array-like): Quadrature nodes.
            - p (int): Classical order.
            - q (int): Stage order.
            - s (int): Number of stages.
    """
    # compute quadrature nodes from right Radau polynomial
    Poly = np.polynomial.Polynomial
    poly = Poly([0, 1]) ** (s - 1) * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # compute coefficent matrix
    V = np.vander(c, increasing=True)
    R = np.diag(1 / np.arange(1, s + 1))
    A = np.diag(c) @ V @ R @ np.linalg.inv(V)

    # extract quadrature weights
    b = A[-1, :]

    # quadrature and stage order
    p = 2 * s - 1
    q = s

    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)
