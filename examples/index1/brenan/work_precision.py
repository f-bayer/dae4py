import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.radau import solve_dae_radau
from dae4py.fortran import dassl, pside, radau, radau5
from brenan import problem


solvers = [
    # (solve_dae_radau, {"s": 3}),
    # (solve_dae_radau, {"s": 5}),
    # (solve_dae_radau, {"s": 7}),
    # TODO: Spot the segfault error
    # (dassl, {}),
    (pside, {}),
    # (radau, {}),
    # (radau5, {}),
]


if __name__ == "__main__":
    # exponents
    # m_max = 45
    m_max = 32
    ms = np.arange(m_max + 1)

    # tolerances and initial step size
    rtols = 10 ** (-(1 + ms / 4))
    atols = rtols
    h0s = 1e-2 * rtols

    # time span
    t_span = problem.t_span
    t0, t1 = t_span

    # initial conditions
    y0 = problem.y0
    yp0 = problem.yp0

    # benchmark results
    results = np.zeros((len(solvers), len(rtols), 2))

    # reference/true solution
    y_ref, yp_ref = problem.true_sol(t1)

    for i, method_and_kwargs in enumerate(solvers):
        method, kwargs = method_and_kwargs
        print(f" - method: {method}; kwargs: {kwargs}")
        for j, (rtol, atol, h0) in enumerate(zip(rtols, atols, h0s)):
            print(f"   * rtol: {rtol}")
            print(f"   * atol: {atol}")
            print(f"   * h0:   {h0}")

            # solve system
            start = time.time()
            sol = method(
                problem.F,
                y0,
                yp0,
                t_span,
                atol=atol,
                rtol=rtol,
                # h0=h0,
                **kwargs,
            )
            end = time.time()
            elapsed_time = end - start
            # print(f"     => sol: {sol}")

            # error
            try:
                diff = y_ref - sol.y[-1]
            except:
                diff = y_ref - sol["y"][-1]
            error = np.linalg.norm(diff)
            print(f"     => error: {error}")

            results[i, j] = (error, elapsed_time)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ri in enumerate(results):
        ax.plot(ri[:, 0], ri[:, 1], label=solvers[i])

    ax.set_title(f"work-precision: brenan")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    ax.set_xlabel("||y_ref(t1) - y(t1)||")
    ax.set_ylabel("elapsed time [s]")

    plt.savefig(f"brenan_work_precision.png", dpi=300)

    plt.show()
