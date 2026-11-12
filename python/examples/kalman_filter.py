"""The Kalman filter (predict and update) with ulm.lam.

Data assimilation for a field on a periodic 1D grid (n points) that is
transported and diffused (advection-diffusion), observed by m sensors at
random grid points.  In every step:

    predict:  x = F x,                      P = F P F^T + Q
    update:   S = H P H^T + R,              K = P H^T S^-1
              x = x + K (z - H x),          P = P - K H P

written in Python as in the textbook.  ulm.lam keeps x and P in place
across the steps, factorises S by Cholesky (asserted spd), solves for K
instead of forming S^-1, evaluates K H P as K (H P) and reuses H P, which
S = (H P) H^T already computed (P is symmetric).  NumPy evaluates the same
formula from left to right: (K H) P costs n^2 m + n^3 instead of n^2 m.
Compared with NumPy as written (inv) and with the version an expert
writes by hand (H P once, Cholesky, P - K (H P)).

    python3 python/examples/kalman_filter.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Index, Matrix, Vector  # noqa: E402

measure = lam.extern("measure", [Index], Vector("m"))
estimate = lam.extern("estimate", [Index, Vector("n")])


@lam.function
def kalman(F: Matrix("n", "n"), Q: Matrix("n", "n", "spsd"),
           H: Matrix("m", "n"), R: Matrix("m", "m", "spd"),
           x0: Vector("n"), P0: Matrix("n", "n", "spd"), steps: Index):
    with lam.loop(steps, carry=[x0, P0]) as L:
        x, P = L.carried
        P = lam.assume(P, "symmetric")      # a covariance stays symmetric
        x = F @ x                                       # predict
        P = F @ P @ F.T + Q
        S = lam.assume(H @ P @ H.T + R, "spd")          # update
        K = P @ H.T @ S.inv
        x = x + K @ (measure(L.index) - H @ x)
        P = P - K @ H @ P
        estimate(L.index, x)
        L.next(x, P)
    return L.results


def numpy_literal(F, Q, H, R, x, P, zs):
    out = []
    for z in zs:
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = P - K @ H @ P
        out.append(x)
    return np.array(out), P


try:
    from scipy.linalg import cho_factor, cho_solve
except ImportError:
    cho_factor = None


def numpy_by_hand(F, Q, H, R, x, P, zs):
    out = []
    for z in zs:
        x = F @ x
        P = F @ P @ F.T + Q
        HP = H @ P
        S = HP @ H.T + R
        if cho_factor:
            K = cho_solve(cho_factor(S, lower=True), HP).T
        else:
            K = np.linalg.solve(S, HP).T
        x = x + K @ (z - H @ x)
        P = P - K @ HP
        out.append(x)
    return np.array(out), P


def model(n, m, steps, rng):
    """Advection-diffusion on a periodic grid (implicit-explicit step as a
    dense matrix), correlated process noise, m random sensors."""
    c, nu = 0.4, 0.2  # Courant number, diffusion number
    i = np.arange(n)
    A = np.zeros((n, n))
    A[i, i] = 1 - 2 * nu
    A[i, (i + 1) % n] = nu - c / 2
    A[i, (i - 1) % n] = nu + c / 2
    F = A
    d = np.minimum(np.abs(i[:, None] - i[None]), n - np.abs(i[:, None] - i[None]))
    Q = 0.01 * np.exp(-d / 10.0)
    sensors = np.sort(rng.choice(n, m, replace=False))
    H = np.zeros((m, n))
    H[np.arange(m), sensors] = 1
    R = 0.05 * np.eye(m)
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(n))
    truth = np.sin(2 * np.pi * i / n) + 0.5 * np.sin(6 * np.pi * i / n)
    zs = []
    for _ in range(steps):
        truth = F @ truth + Lq @ rng.normal(size=n)
        zs.append(H @ truth + np.sqrt(0.05) * rng.normal(size=m))
    return F, Q, H, R, np.zeros(n), np.eye(n), np.array(zs), truth


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    lib = lam.compile(kalman, name="kalman_filter_py")
    # the generated code, kept in the repository (doc/ulm-lam.tex, appendix)
    (Path(__file__).resolve().parent / "generated" / "kalman_filter.cpp").write_text(
        lib.source)
    rng = np.random.default_rng(5)
    print("Kalman filter, advection-diffusion on a periodic grid; times in s,"
          " (x): relative to ulm.lam")
    print("     n     m  steps   ulm.lam   NumPy as written   NumPy by hand"
          "     difference")
    ok = True
    for n, m, steps in ((500, 50, 50), (1000, 100, 50), (2000, 200, 20)):
        F, Q, H, R, x0, P0, zs, truth = model(n, m, steps, rng)
        X = np.empty((steps, n))

        def rec(i, x):
            X[i] = x[:, 0]

        lib.bind(measure=lambda i: zs[i], estimate=rec)
        reps = 3 if n <= 1000 else 1
        t_lam, (xl, Pl) = best(lambda: lib.kalman(F, Q, H, R, x0, P0, steps),
                               reps)
        t_lit, _ = best(lambda: numpy_literal(F, Q, H, R, x0, P0, zs), 1)
        t_hand, (Xh, Ph) = best(lambda: numpy_by_hand(F, Q, H, R, x0, P0, zs),
                                reps)
        err = max(np.abs(X - Xh).max() / np.abs(Xh).max(),
                  np.abs(Pl - Ph).max() / np.abs(Ph).max())
        ok &= err < 1e-9
        print(f"  {n:4d}  {m:4d}  {steps:5d}   {t_lam:7.3f}   {t_lit:7.3f} "
              f"({t_lit / t_lam:4.1f}x)   {t_hand:7.3f} ({t_hand / t_lam:4.1f}x)"
              f"   {err:.1e}")
    rmse = np.sqrt(np.mean((X[-1] - truth) ** 2))
    print(f"error of the last estimate (n = {n}): RMSE {rmse:.3f}, the field "
          f"varies by {np.ptp(truth):.1f}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
