"""The Laplace approximation of a last layer: one formula, two algorithms.

A network with features phi(x) (here random ReLU features of a two-layer
network) gets a Bayesian last layer: with N training points and p features,
Phi (N x p), prior precision alpha and noise precision beta, the posterior
of the last layer's weights is Gaussian with precision

    A = alpha I + beta Phi^T Phi                       (p x p)

and the prediction at M test points Phi* (M x p) has

    mean = beta Phi* A^-1 Phi^T y,   cov = Phi* A^-1 Phi*^T.

This is Bayesian linear regression, and the last-layer Laplace approximation
of e.g. laplace-torch for regression.  Which algorithm is cheaper depends on
N and p.  With many data (N > p) the textbook way: form A (syrk), Cholesky
(p^3 / 3).  With few data and wide features (N < p, e.g. a last layer of
width 4096 fitted to 1000 points) the Woodbury identity

    A^-1 = (1/alpha) (I - beta Phi^T (alpha I + beta Phi Phi^T)^-1 Phi)

factorises an N x N matrix instead, and the products are applied to its
terms, so that no p x p matrix is formed at all.  Libraries choose by hand;
ulm.lam generates both from the formula and chooses at run time (N < p),
as an scf.if around the whole computation.

The comparison: NumPy as written (np.linalg.inv), SciPy written by an
expert in both ways (cho_factor and triangular solves), on the same BLAS
(OpenBLAS, as NumPy and SciPy; ulm.lam with backend="cblas").

    python3 python/examples/laplace.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Float, Matrix, Vector  # noqa: E402


@lam.function
def last_layer(Phi: Matrix("N", "p"), y: Vector("N"), Phis: Matrix("M", "p"),
               alpha: Float, beta: Float):
    A = lam.assume(alpha * lam.eye("p") + beta * (Phi.T @ Phi), "spd")
    return beta * Phis @ A.inv @ Phi.T @ y, Phis @ A.inv @ Phis.T


def numpy_as_written(Phi, y, Phis, alpha, beta):
    Ai = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)
    return beta * Phis @ Ai @ Phi.T @ y, Phis @ Ai @ Phis.T


def scipy_direct(Phi, y, Phis, alpha, beta):
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    A = beta * (Phi.T @ Phi)
    A[np.diag_indices_from(A)] += alpha
    c = cho_factor(A, lower=True)
    mean = beta * (Phis @ cho_solve(c, Phi.T @ y))
    V = solve_triangular(c[0], Phis.T, lower=True)
    return mean, V.T @ V


def scipy_woodbury(Phi, y, Phis, alpha, beta):
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    K = beta * (Phi @ Phi.T)
    K[np.diag_indices_from(K)] += alpha
    c = cho_factor(K, lower=True)
    u = Phi.T @ y
    mean = (beta / alpha) * (Phis @ u - beta * (Phis @ (Phi.T @ cho_solve(
        c, Phi @ u))))
    V = solve_triangular(c[0], Phi @ Phis.T, lower=True)
    return mean, (Phis @ Phis.T - beta * (V.T @ V)) / alpha


def features(X, W1, b1):
    return np.maximum(X @ W1 + b1, 0.0) / np.sqrt(W1.shape[1])


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    libs = {"OpenBLAS": lam.compile(last_layer, name="laplace_cblas",
                                    backend="cblas"),
            "ulmBLAS": lam.compile(last_layer, name="laplace_ulm")}
    print(libs["ulmBLAS"].last_layer.code)
    try:
        import scipy  # noqa: F401
        have_scipy = True
    except ImportError:
        have_scipy = False
    rng = np.random.default_rng(5)
    d, M, alpha, beta = 8, 1000, 1.0, 100.0
    print("last-layer Laplace: predictive mean and covariance at M = "
          f"{M} test points, float64; seconds, (x): relative to ulm.lam "
          "on OpenBLAS")
    print("      N      p   ulm.lam/OpenBLAS  ulm.lam/ulmBLAS   NumPy (inv) "
          "      SciPy direct     SciPy Woodbury   difference")
    ok = True
    for N, p in ((20000, 500), (4000, 2000), (2000, 4000), (1000, 8000)):
        W1, b1 = rng.normal(size=(d, p)), rng.normal(size=p)
        X, Xs = rng.normal(size=(N, d)), rng.normal(size=(M, d))
        f = lambda x: np.sin(x @ np.arange(1, d + 1)) / 2  # noqa: E731
        y = f(X) + 0.1 * rng.normal(size=N)
        Phi, Phis = features(X, W1, b1), features(Xs, W1, b1)
        reps = 3
        times, res = {}, {}
        for name, lib in libs.items():
            times[name], r = best(
                lambda: lib.last_layer(Phi, y, Phis, alpha, beta), reps)
            res[name] = (r[0].ravel(), r[1])
        times["numpy"], res["numpy"] = best(
            lambda: numpy_as_written(Phi, y, Phis, alpha, beta), reps)
        if have_scipy:
            times["direct"], res["direct"] = best(
                lambda: scipy_direct(Phi, y, Phis, alpha, beta), reps)
            times["woodbury"], res["woodbury"] = best(
                lambda: scipy_woodbury(Phi, y, Phis, alpha, beta), reps)
        ref = res["direct" if have_scipy else "numpy"]
        err = max(max(np.abs(m - ref[0]).max() / np.abs(ref[0]).max(),
                      np.abs(c - ref[1]).max() / np.abs(ref[1]).max())
                  for k, (m, c) in res.items() if k != "numpy")
        ok &= err < 1e-8
        t0 = times["OpenBLAS"]

        def fmt(k):
            t = times.get(k)
            return "        -       " if t is None else \
                f"{t:7.3f} ({t / t0:4.1f}x)"

        print(f"  {N:5d}  {p:5d}   {t0:9.3f}        {fmt('ulmBLAS')}  "
              f"{fmt('numpy')}  {fmt('direct')}  {fmt('woodbury')}   "
              f"{err:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
