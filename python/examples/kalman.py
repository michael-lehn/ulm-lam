"""Formula (1) in a loop, written in Python (examples/kalman/estimate.mlir).

A sensor array (n sensors, m states) with noise covariance R delivers a
measurement y_i in every step; the estimate is

    b_i = (H^T R^-1 H)^-1 H^T R^-1 y_i.

The program is written in Python as on paper; ulm.lam compiles it to C++
calling ulmBLAS (factorisations before the loop, O(n^2) per step).  The
measurements come from a Python function, the estimates go to one
(observe and record are callbacks).  Compared with NumPy (OpenBLAS): the
formula written literally in the loop, and the operator
G = (H^T R^-1 H)^-1 H^T R^-1 computed by hand before the loop.

    python3 python/examples/kalman.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Index, Matrix, Vector  # noqa: E402

n, m, steps = 4000, 200, 200

observe = lam.extern("observe", [Index], Vector(n))
record = lam.extern("record", [Index, Vector(m)])


@lam.function
def estimate(H: Matrix(n, m, "full_rank"), R: Matrix(n, n, "spd"), steps: Index):
    with lam.loop(steps) as L:
        y = observe(L.index)
        b = (H.T @ R.inv @ H).inv @ H.T @ R.inv @ y
        record(L.index, b)


def best(f, reps=3):
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t


def main():
    t0 = time.perf_counter()
    lib = lam.compile(estimate, name="kalman_py")
    # the generated code, kept in the repository (doc/ulm-lam.tex, appendix)
    (Path(__file__).resolve().parent / "generated" / "kalman.cpp").write_text(
        lib.source)
    print(
        f"compiled in {time.perf_counter() - t0:.1f} s (cached when "
        f"unchanged): {lib.path}"
    )
    print("generated C++ (lib.estimate.cpp):\n")
    print(lib.estimate.cpp)

    rng = np.random.default_rng(3)
    H = rng.uniform(-1, 1, (n, m))
    A = rng.uniform(-1, 1, (n, n))
    R = A @ A.T / n + np.eye(n)  # correlated noise, spd
    xs = np.sin(0.1 * np.arange(steps)[:, None] + 0.05 * np.arange(m))
    ys = xs @ H.T + 0.1 * rng.uniform(-1, 1, (steps, n))

    B = np.empty((steps, m))

    def rec(i, b):
        B[i] = b[:, 0]  # b is a view of lamp's memory: copy it

    lib.bind(observe=lambda i: ys[i], record=rec)
    t_lam = best(lambda: lib.estimate(H, R, steps))

    def literal():  # the formula as written, in every step
        out = np.empty((steps, m))
        for i in range(steps):
            Ri = np.linalg.inv(R)
            out[i] = np.linalg.inv(H.T @ Ri @ H) @ H.T @ Ri @ ys[i]
        return out

    def hoisted():  # by hand: G once, one product per step
        RiH = np.linalg.solve(R, H)
        G = np.linalg.solve(H.T @ RiH, RiH.T)
        out = np.empty((steps, m))
        for i in range(steps):
            out[i] = G @ ys[i]
        return out

    t_lit = best(literal, 1)
    t_hoist = best(hoisted)
    ref = hoisted()
    err = np.abs(B - ref).max() / np.abs(ref).max()

    print(f"{steps} steps, n = {n}, m = {m}")
    print(f"  ulm.lam (generated C++, ulmBLAS)   {1e3 * t_lam:8.1f} ms")
    print(f"  NumPy, formula as written          {1e3 * t_lit:8.1f} ms")
    print(f"  NumPy, operator G hoisted by hand  {1e3 * t_hoist:8.1f} ms")
    print(f"relative difference to NumPy: {err:.1e}")
    print("PASS" if err < 1e-10 else "FAIL")
    return 0 if err < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
