"""Bayesian linear regression with ulm.lam.

With the prior w ~ N(0, alpha^-1 I) and noise precision beta, the
posterior of the weights and the prediction at test points Phi* are

    A = alpha I + beta Phi^T Phi,       w = beta A^-1 Phi^T t
    mean = Phi* w,                      cov = Phi* A^-1 Phi*^T + beta^-1 I

(Bishop, Pattern Recognition and Machine Learning, Sec. 3.3), written in
Python with alpha and beta as Float parameters.  On the bike sharing data
with random Fourier features (as ridge.py): the predictive mean and the
full predictive covariance of all test hours.

ulm.lam: Phi^T Phi by syrk, scaled by beta and alpha added to its diagonal
(one triangle), Cholesky; the mean by two triangular solves and two gemv,
beta applied to the shortest vector; the covariance as V^T V with
V = L^-1 Phi*^T (trsm, syrk), 1/beta added to its diagonal.  Compared with
NumPy as written (inv) and with the version written by hand (SciPy:
Cholesky, solves, V^T V).

    python3 python/examples/bayes.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ulm import lam  # noqa: E402
from ulm.lam import Float, Matrix, Vector  # noqa: E402
from gp import bike_data  # noqa: E402


@lam.function
def blr(Phi: Matrix("n", "p"), t: Vector("n"), Phis: Matrix("m", "p"),
        alpha: Float, beta: Float):
    A = lam.assume(alpha * lam.eye("p") + beta * (Phi.T @ Phi), "spd")
    w = beta * A.inv @ Phi.T @ t
    return Phis @ w, Phis @ A.inv @ Phis.T + (1 / beta) * lam.eye("m")


def numpy_literal(Phi, t, Phis, alpha, beta):
    S = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)
    w = beta * S @ Phi.T @ t
    return Phis @ w, Phis @ S @ Phis.T + np.eye(len(Phis)) / beta


try:
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    def by_hand(Phi, t, Phis, alpha, beta):
        A = beta * (Phi.T @ Phi)
        A[np.diag_indices_from(A)] += alpha
        L = np.linalg.cholesky(A)
        w = beta * cho_solve((L, True), Phi.T @ t)
        V = solve_triangular(L, Phis.T, lower=True)
        cov = V.T @ V
        cov[np.diag_indices_from(cov)] += 1 / beta
        return Phis @ w, cov
except ImportError:
    def by_hand(Phi, t, Phis, alpha, beta):
        A = alpha * np.eye(Phi.shape[1]) + beta * (Phi.T @ Phi)
        L = np.linalg.cholesky(A)
        w = beta * np.linalg.solve(A, Phi.T @ t)
        V = np.linalg.solve(L, Phis.T)
        return Phis @ w, V.T @ V + np.eye(len(Phis)) / beta


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    lib = lam.compile(blr, name="bayes_py")
    # the generated code, kept in the repository (doc/ulm-lam.tex, appendix)
    (Path(__file__).resolve().parent / "generated" / "bayes.cpp").write_text(
        lib.source)
    X0, y0 = bike_data()
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(y0))
    test, train = perm[:1379], perm[1379:]
    alpha, beta = 1.0, 10.0
    print(f"Bike sharing: {len(train)} training hours, {len(test)} test hours "
          f"(mean and full covariance), alpha = {alpha}, beta = {beta}")
    print("  seconds, (x): relative to ulm.lam")
    print("     p   RMSE  in 2 sd   ulm.lam   NumPy as written   by hand "
          "(SciPy)   difference")
    ok = True
    for p in (500, 1000, 2000, 4000):
        W = rng.normal(0, 1 / 2.0, (X0.shape[1], p))
        b = rng.uniform(0, 2 * np.pi, p)
        Z = np.sqrt(2 / p) * np.cos(X0 @ W + b)
        Phi, t, Phis = Z[train], y0[train].copy(), Z[test]
        reps = 3 if p <= 2000 else 1
        t_lam, (mean, cov) = best(lambda: lib.blr(Phi, t, Phis, alpha, beta),
                                  reps)
        t_lit, _ = best(lambda: numpy_literal(Phi, t, Phis, alpha, beta), 1)
        t_hand, (rm, rc) = best(lambda: by_hand(Phi, t, Phis, alpha, beta),
                                reps)
        err = max(np.abs(mean[:, 0] - rm).max() / np.abs(rm).max(),
                  np.abs(cov - rc).max() / np.abs(rc).max())
        ok &= err < 1e-8
        e = mean[:, 0] - y0[test]
        inside = np.mean(np.abs(e) < 2 * np.sqrt(np.diag(cov)))
        print(f"  {p:4d}  {np.sqrt(np.mean(e**2)):5.3f}  {100 * inside:5.0f} %"
              f"   {t_lam:7.3f}   {t_lit:7.3f} ({t_lit / t_lam:4.1f}x)   "
              f"{t_hand:7.3f} ({t_hand / t_lam:4.1f}x)   {err:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
