"""Formula (1) in a loop with a covariance that changes in every step
(examples/kalman/adaptive.mlir), written in Python.

The noise covariance is re-estimated in every step (an adaptive filter, or
sensors whose noise drifts):

    R_i = update_covariance(i, R_{i-1})          n x n, spd
    b_i = (H^T R_i^-1 H)^-1 H^T R_i^-1 y_i

update_covariance is an external function that updates R in place (it is
declared with updates=1: its result is the new value of parameter 1).  Here
it is a Python callback that changes R through a NumPy view of ulm.lam's
memory.  The Cholesky factorisation of R_i now has to run in every step;
ulm.lam shares it between both occurrences of R_i^-1 and reuses
W = L_i^-1 H for H^T R_i^-1 y_i.  Compared with NumPy: the formula as
written, and the best one can do by hand in NumPy when R changes (one LU
solve per step for [H, y_i]).

    python3 python/examples/adaptive_kalman.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Index, Matrix, Vector  # noqa: E402

n, m, steps = 4000, 200, 10

update_covariance = lam.extern("update_covariance",
                               [Index, Matrix(n, n, "spd")],
                               Matrix(n, n, "spd"), updates=1)
observe = lam.extern("observe", [Index], Vector(n))
record = lam.extern("record", [Index, Vector(m)])


@lam.function
def estimate_adaptive(H: Matrix(n, m, "full_rank"), R0: Matrix(n, n, "spd"),
                      steps: Index):
    with lam.loop(steps, carry=[R0]) as L:
        (R,) = L.carried
        R = update_covariance(L.index, R)
        y = observe(L.index)
        b = (H.T @ R.inv @ H).inv @ H.T @ R.inv @ y
        record(L.index, b)
        L.next(R)


def main():
    t0 = time.perf_counter()
    lib = lam.compile(estimate_adaptive, name="adaptive_kalman_py")
    print(f"compiled in {time.perf_counter() - t0:.1f} s (cached when "
          f"unchanged): {lib.path}")
    print("generated C++ (lib.estimate_adaptive.cpp):\n")
    print(lib.estimate_adaptive.cpp)

    rng = np.random.default_rng(5)
    H = rng.uniform(-1, 1, (n, m))
    A = rng.uniform(-1, 1, (n, n))
    Rbase = A @ A.T / n + np.eye(n)
    xs = np.sin(0.1 * np.arange(steps)[:, None] + 0.05 * np.arange(m))
    ys = xs @ H.T + 0.1 * rng.uniform(-1, 1, (steps, n))
    diag = np.arange(n)

    def update(i, R):
        """R <- 0.9 R + 0.1 R_base + 0.05 (1 + sin(0.3 i)) I, in place."""
        R *= 0.9
        R += 0.1 * Rbase
        R[diag, diag] += 0.05 * (1 + np.sin(0.3 * i))

    B = np.empty((steps, m))

    def rec(i, b):
        B[i] = b[:, 0]

    lib.bind(update_covariance=update, observe=lambda i: ys[i], record=rec)
    R0 = Rbase.copy()
    t0 = time.perf_counter()
    lib.estimate_adaptive(H, R0, steps)
    t_lam = time.perf_counter() - t0
    unchanged = np.array_equal(R0, Rbase)  # R0 is an in parameter

    def literal():  # the formula as written, explicit inverses
        R, out = Rbase.copy(), np.empty((steps, m))
        for i in range(steps):
            update(i, R)
            Ri = np.linalg.inv(R)
            out[i] = np.linalg.inv(H.T @ Ri @ H) @ H.T @ Ri @ ys[i]
        return out

    def by_hand():  # one LU solve per step for [H, y_i]
        R, out = Rbase.copy(), np.empty((steps, m))
        for i in range(steps):
            update(i, R)
            Z = np.linalg.solve(R, np.column_stack([H, ys[i]]))
            out[i] = np.linalg.solve(H.T @ Z[:, :m], H.T @ Z[:, m])
        return out

    t0 = time.perf_counter()
    lit = literal()
    t_lit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = by_hand()
    t_hand = time.perf_counter() - t0
    err = max(np.abs(B - ref).max(), np.abs(lit - ref).max()) / \
        np.abs(ref).max()

    print(f"{steps} steps, n = {n}, m = {m}, R updated in every step")
    print(f"  ulm.lam (generated C++, ulmBLAS)   {1e3 * t_lam:8.1f} ms")
    print(f"  NumPy, formula as written          {1e3 * t_lit:8.1f} ms")
    print(f"  NumPy, one LU solve per step       {1e3 * t_hand:8.1f} ms")
    print(f"relative difference to NumPy: {err:.1e}; R0 unchanged: "
          f"{unchanged}")
    ok = err < 1e-10 and unchanged
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
