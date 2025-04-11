import numpy as np
import pytest
from dae4py.butcher_tableau import gauss_legendre_tableau, radau_tableau


@pytest.mark.parametrize("s,", [1, 2, 3])
def test_gauss_tableau(s):
    match s:
        case 1:
            A_desired = np.array([[1 / 2]])
            b_desired = np.array([1.0])
        case 2:
            S3 = np.sqrt(3)
            A_desired = np.array(
                [
                    [1 / 4, 1 / 4 - S3 / 6],
                    [1 / 4 + S3 / 6, 1 / 4],
                ]
            )
            b_desired = np.array([1 / 2, 1 / 2])
        case 3:
            S15 = np.sqrt(15)
            A_desired = np.array(
                [
                    [
                        5 / 36,
                        2 / 9 - S15 / 15,
                        5 / 36 - S15 / 30,
                    ],
                    [
                        5 / 36 + S15 / 24,
                        2 / 9,
                        5 / 36 - S15 / 24,
                    ],
                    [
                        5 / 36 + S15 / 30,
                        2 / 9 + S15 / 15,
                        5 / 36,
                    ],
                ]
            )
            b_desired = np.array([5 / 18, 4 / 9, 5 / 18])
        case _:
            raise NotImplementedError

    p_desired = 2 * s
    q_desired = s
    c_desired = np.sum(A_desired, axis=1)

    tab = gauss_legendre_tableau(s)

    assert np.allclose(tab.A, A_desired)
    assert np.allclose(tab.b, b_desired)
    assert np.allclose(tab.c, c_desired)
    assert np.allclose(tab.p, p_desired)
    assert np.allclose(tab.q, q_desired)


@pytest.mark.parametrize("s,", [1, 2, 3])
def test_radau_tableau(s):
    match s:
        case 1:
            A_desired = np.array([[1.0]])
        case 2:
            A_desired = np.array(
                [
                    [5 / 12, -1 / 12],
                    [3 / 4, 1 / 4],
                ]
            )
        case 3:
            S6 = np.sqrt(6)
            A_desired = np.array(
                [
                    [
                        11 / 45 - 7 * S6 / 360,
                        37 / 225 - 169 * S6 / 1800,
                        -2 / 225 + S6 / 75,
                    ],
                    [
                        37 / 225 + 169 * S6 / 1800,
                        11 / 45 + 7 * S6 / 360,
                        -2 / 225 - S6 / 75,
                    ],
                    [4 / 9 - S6 / 36, 4 / 9 + S6 / 36, 1 / 9],
                ]
            )
        case _:
            raise NotImplementedError

    p_desired = 2 * s - 1
    q_desired = s
    b_desired = A_desired[-1, :]
    c_desired = np.sum(A_desired, axis=1)

    tab = radau_tableau(s)

    assert np.allclose(tab.A, A_desired)
    assert np.allclose(tab.b, b_desired)
    assert np.allclose(tab.c, c_desired)
    assert np.allclose(tab.p, p_desired)
    assert np.allclose(tab.q, q_desired)
