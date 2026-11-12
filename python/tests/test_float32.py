"""Float32: lam.compile(..., dtype="float32") compiles the same kernels
on float.  Needs NumPy.  Run:  python3 -m unittest discover -s python/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Float, Matrix, Vector  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@lam.function
def f32_gls(X: Matrix("n", "m", "full_rank"), M: Matrix("n", "n", "spd"),
            y: Vector("n")):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y


@lam.function
def f32_ops(A: Matrix("n", "n"), x: Vector("n"), a: Float):
    return a * (A @ x) - A.inv @ x / a


@unittest.skipIf(np is None, "needs NumPy")
class Float32(unittest.TestCase):
    def test_float32(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 4))
        A = rng.normal(size=(80, 80))
        M = A @ A.T + 80 * np.eye(80)
        y = rng.normal(size=80)
        B = A + 20 * np.eye(80)
        ref = np.linalg.solve(X.T @ np.linalg.solve(M, X),
                              X.T @ np.linalg.solve(M, y))
        ref2 = 1.5 * (B @ y) - np.linalg.solve(B, y) / 1.5
        lib = lam.compile([f32_gls, f32_ops], name="f32_test_py",
                          dtype="float32")
        f = np.float32
        r = lib.f32_gls(X.astype(f), M.astype(f), y.astype(f))
        self.assertEqual(r.dtype, np.float32)
        np.testing.assert_allclose(r.ravel(), ref, rtol=1e-3, atol=1e-4)
        # a float64 input is converted for reading
        r2 = lib.f32_ops(B, y.astype(f), 1.5)
        self.assertEqual(r2.dtype, np.float32)
        np.testing.assert_allclose(r2.ravel(), ref2, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()


class Code(unittest.TestCase):
    """lib.f.code: the derivation (Julia's @code_lam)."""

    def test_code(self):
        lib = lam.compile([f32_gls], name="code_test_py")
        t = lib.f32_gls.code
        self.assertIn("potrf    L1 ← chol(M)", t)
        self.assertIn("syrk", t)
        self.assertNotIn("%", t)
        self.assertIn("info 1:  Cholesky factorisation of M", t)
