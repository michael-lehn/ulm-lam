"""Ridge regression with ulm.lam: (X^T X + lambda I)^-1 X^T Y.

On the bike sharing data of gp.py (UCI, University of California, Irvine):
random Fourier features (Rahimi and Recht, Random features for large-scale
kernel machines, NIPS 2007) turn the 14 features of each hour into p
features whose ridge regression approximates kernel ridge regression with
an RBF (radial basis function) kernel, for all n = 16000 training hours.

ulm.lam: X^T X by syrk (one triangle), lambda added to its diagonal,
Cholesky factorisation of the triangle, and the order of (X^T X + lambda
I)^-1 X^T Y chosen at run time from n, p and the number of targets k.
Compared with NumPy as written (inv), NumPy by hand (np.linalg.solve),
SciPy by hand (cho_factor, cho_solve) and scikit-learn's Ridge with the
Cholesky solver.

    python3 python/examples/ridge.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ulm import lam  # noqa: E402
from ulm.lam import Float, Matrix  # noqa: E402
from gp import bike_data  # noqa: E402


@lam.function
def ridge(X: Matrix("n", "p"), Y: Matrix("n", "k"), lam_: Float):
    A = lam.assume(X.T @ X + lam_ * lam.eye("p"), "spd")
    return A.inv @ X.T @ Y


def numpy_literal(X, Y, lam_):
    return np.linalg.inv(X.T @ X + lam_ * np.eye(X.shape[1])) @ X.T @ Y


def numpy_by_hand(X, Y, lam_):
    return np.linalg.solve(X.T @ X + lam_ * np.eye(X.shape[1]), X.T @ Y)


try:
    from scipy.linalg import cho_factor, cho_solve

    def scipy_by_hand(X, Y, lam_):
        A = X.T @ X
        A[np.diag_indices_from(A)] += lam_
        return cho_solve(cho_factor(A, lower=True), X.T @ Y)
except ImportError:
    scipy_by_hand = None

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    lib = lam.compile(ridge, name="ridge_py")
    # the generated code, kept in the repository (doc/ulm-lam.tex, appendix)
    (Path(__file__).resolve().parent / "generated" / "ridge.cpp").write_text(
        lib.source)
    X0, y0 = bike_data()
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(y0))
    test, train = perm[:1379], perm[1379:]
    lam_ = 1.0
    print(f"Bike sharing: {len(train)} training hours, {len(test)} test hours,"
          " random Fourier features, lambda = 1")
    print("  times in s, (x): relative to ulm.lam; RMSE in standard "
          "deviations of log(1 + rentals)")
    print("     p   RMSE   ulm.lam   NumPy literal  NumPy by hand   SciPy by "
          "hand   scikit-learn  difference")
    ok = True
    for p in (500, 1000, 2000, 4000):
        W = rng.normal(0, 1 / 2.0, (X0.shape[1], p))
        b = rng.uniform(0, 2 * np.pi, p)
        Z = np.sqrt(2 / p) * np.cos(X0 @ W + b)
        X, Y = Z[train], y0[train, None].copy()
        reps = 3 if p <= 2000 else 1
        t_lam, beta = best(lambda: lib.ridge(X, Y, lam_), reps)
        t_lit, _ = best(lambda: numpy_literal(X, Y, lam_), 1)
        t_np, ref = best(lambda: numpy_by_hand(X, Y, lam_), reps)
        t_sp = (best(lambda: scipy_by_hand(X, Y, lam_), reps)[0]
                if scipy_by_hand else float("nan"))
        t_sk = float("nan")
        if Ridge:
            m = Ridge(alpha=lam_, fit_intercept=False, solver="cholesky")
            t_sk, _ = best(lambda: m.fit(X, Y[:, 0]), reps)
        err = np.abs(beta - ref).max() / np.abs(ref).max()
        ok &= err < 1e-8
        rmse = np.sqrt(np.mean((Z[test] @ beta[:, 0] - y0[test]) ** 2))

        def f(t):
            return "      -      " if t != t else f"{t:6.3f} ({t / t_lam:4.1f}x)"

        print(f"  {p:4d}  {rmse:5.3f}  {t_lam:7.3f}   {f(t_lit)}  {f(t_np)}"
              f"  {f(t_sp)}  {f(t_sk)}  {err:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
