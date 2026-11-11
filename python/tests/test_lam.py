"""Tests of the Python frontend of ulm.lam.

Every compiled function is compared with a pure-Python evaluation at small
sizes.  Arguments are lists, array.array buffers and (if installed) NumPy
arrays; results are NumPy arrays if NumPy is installed, flat array.array
buffers otherwise.  Run:  python3 -m unittest discover -s python/tests
"""

import random
import sys
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Index, InOut, LamError, Matrix, Vector  # noqa: E402

N, K = 40, 6
rnd = random.Random(4)


# ---- pure-Python reference ------------------------------------------------

def rand(r, c):
    return [[rnd.uniform(-1, 1) for _ in range(c)] for _ in range(r)]


def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)]
            for row in A]


def tr(A):
    return [list(c) for c in zip(*A)]


def add(A, B, alpha=1.0):
    return [[a + alpha * b for a, b in zip(r, s)] for r, s in zip(A, B)]


def solve(A, B):
    """A^-1 B by Gaussian elimination with partial pivoting."""
    A = [r[:] for r in A]
    B = [r[:] for r in B]
    n = len(A)
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(A[i][j]))
        A[j], A[p], B[j], B[p] = A[p], A[j], B[p], B[j]
        for i in range(j + 1, n):
            f = A[i][j] / A[j][j]
            A[i] = [a - f * b for a, b in zip(A[i], A[j])]
            B[i] = [a - f * b for a, b in zip(B[i], B[j])]
    for j in reversed(range(n)):
        B[j] = [(B[j][q] - sum(A[j][i] * B[i][q] for i in range(j + 1, n)))
                / A[j][j] for q in range(len(B[j]))]
    return B


def spd(n):
    A = rand(n, n)
    return add(matmul(A, tr(A)), [[n if i == j else 0 for j in range(n)]
                                   for i in range(n)])


def flat(M):
    return [v for row in M for v in row]


def values(x):
    """Flat list of a result (NumPy array, array.array, list of lists)."""
    if hasattr(x, "ravel"):
        return x.ravel().tolist()
    if x and isinstance(x[0], list):
        return flat(x)
    return list(x)


def rel(x, ref):
    x, ref = values(x), values(ref)
    assert len(x) == len(ref)
    return max(abs(a - b) for a, b in zip(x, ref)) / max(abs(b) for b in ref)


# ---- the programs -----------------------------------------------------------

@lam.function
def gls(X: Matrix(N, K, "full_rank"), M: Matrix(N, N, "spd"), y: Vector(N)):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y


@lam.function
def gls_inout(X: Matrix(N, K, "full_rank"), M: InOut(Matrix(N, N, "spd")),
              y: InOut(Vector(N))):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y


foo = lam.extern("foo", [Index], Matrix(K, K))


@lam.function
def accumulate(A: Matrix(K, K), C: Matrix(K, K), X: InOut(Matrix(K, K)),
               n: Index):
    with lam.loop(n, carry=[X]) as L:
        (Xi,) = L.carried
        B = foo(L.index)
        L.next(Xi + A @ B @ C)
    lam.update(X, L.results[0])


observe = lam.extern("observe", [Index], Vector(N))
record = lam.extern("record", [Index, Vector(K)])


@lam.function
def estimate(H: Matrix(N, K, "full_rank"), R: Matrix(N, N, "spd"), n: Index):
    with lam.loop(n) as L:
        y = observe(L.index)
        b = (H.T @ R.inv @ H).inv @ H.T @ R.inv @ y
        record(L.index, b)


update_cov = lam.extern("update_cov", [Index, Matrix(N, N, "spd")],
                        Matrix(N, N, "spd"), updates=1)


@lam.function
def adaptive(H: Matrix(N, K, "full_rank"), R0: Matrix(N, N, "spd"),
             n: Index):
    # the covariance is carried and updated in place by an external function
    with lam.loop(n, carry=[R0]) as L:
        (R,) = L.carried
        R = update_cov(L.index, R)
        y = observe(L.index)
        record(L.index, (H.T @ R.inv @ H).inv @ H.T @ R.inv @ y)
        L.next(R)


@lam.function
def twice(X: Matrix(N, K, "full_rank"), M: Matrix(N, N, "spd"), y: Vector(N)):
    # a function calling a function (workspaces accumulate)
    return gls(X, M, y) + 2.0 * gls(X, M, y)


@lam.function
def combos(A: Matrix(K, K), B: Matrix(K, K), L: Matrix(K, K, "lower_tri")):
    return A.inv @ B - 0.5 * B @ L.inv.T, A.T @ A


class Lam(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lib = lam.compile([gls, gls_inout, accumulate, estimate, adaptive,
                               twice, combos], name="tests")
        cls.X, cls.M, cls.y = rand(N, K), spd(N), [rnd.uniform(-1, 1)
                                                   for _ in range(N)]
        Y = [[v] for v in cls.y]
        MiX, Miy = solve(cls.M, cls.X), solve(cls.M, Y)
        cls.b = flat(solve(matmul(tr(cls.X), MiX), matmul(tr(cls.X), Miy)))

    def test_gls(self):
        b = self.lib.gls(self.X, self.M, self.y)
        self.assertLess(rel(b, self.b), 1e-12)

    def test_gls_buffers(self):
        b = self.lib.gls(array("d", flat(self.X)), array("d", flat(self.M)),
                         array("d", self.y))
        self.assertLess(rel(b, self.b), 1e-12)

    def test_inout_may_overwrite(self):
        M, y = array("d", flat(self.M)), array("d", self.y)
        X = array("d", flat(self.X))
        b = self.lib.gls_inout(X, M, y)
        self.assertLess(rel(b, self.b), 1e-12)
        self.assertEqual(values(X), flat(self.X))  # in: unchanged
        self.assertNotEqual(values(M), flat(self.M))  # inout: overwritten
        self.assertLess(self.lib.gls_inout.workspace_size(X, M, y),
                        self.lib.gls.workspace_size(X, M, y))

    def test_loop_carried_update_and_callback(self):
        A, C, X0 = rand(K, K), rand(K, K), rand(K, K)
        Bs = [rand(K, K) for _ in range(3)]
        self.lib.bind(foo=lambda i: Bs[i])
        X = array("d", flat(X0))
        self.lib.accumulate(A, C, X, 3)
        ref = X0
        for B in Bs:
            ref = add(ref, matmul(matmul(A, B), C))
        self.assertLess(rel(X, ref), 1e-12)

    def test_loop_with_callbacks(self):
        ys = [[rnd.uniform(-1, 1) for _ in range(N)] for _ in range(3)]
        got = {}
        self.lib.bind(observe=lambda i: ys[i],
                      record=lambda i, b: got.__setitem__(i, flat(
                          [[b[r, 0]] for r in range(K)])))
        self.lib.estimate(self.X, self.M, 3)
        for i in range(3):
            lib_b = self.lib.gls(self.X, self.M, ys[i])
            self.assertLess(rel(got[i], lib_b), 1e-12)

    def test_covariance_updated_in_place(self):
        ys = [[rnd.uniform(-1, 1) for _ in range(N)] for _ in range(3)]
        got = {}

        def upd(i, R):  # R is a view of the generated code's memory
            for r in range(N):
                R[r, r] = R[r, r] + 1.0 + i

        self.lib.bind(update_cov=upd, observe=lambda i: ys[i],
                      record=lambda i, b: got.__setitem__(
                          i, [b[r, 0] for r in range(K)]))
        R0 = array("d", flat(self.M))
        self.lib.adaptive(self.X, R0, 3)
        self.assertEqual(values(R0), flat(self.M))  # in: unchanged
        R = [row[:] for row in self.M]
        for i in range(3):
            for r in range(N):
                R[r][r] += 1.0 + i
            ref = self.lib.gls(self.X, R, ys[i])
            self.assertLess(rel(got[i], ref), 1e-12)

    def test_callback_exception_is_raised(self):
        def bad(i):
            raise RuntimeError("from the callback")
        self.lib.bind(observe=bad, record=lambda i, b: None)
        with self.assertRaisesRegex(RuntimeError, "from the callback"):
            self.lib.estimate(self.X, self.M, 1)

    def test_function_calls_function(self):
        b = self.lib.twice(self.X, self.M, self.y)
        self.assertLess(rel(b, [3 * v for v in self.b]), 1e-12)
        self.assertGreater(self.lib.twice.workspace_size(self.X, self.M,
                                                         self.y),
                           self.lib.gls.workspace_size(self.X, self.M,
                                                       self.y))

    def test_two_results(self):
        A, B = rand(K, K), rand(K, K)
        Lt = rand(K, K)
        L = [[Lt[i][j] if j <= i else 0.0 for j in range(K)]
             for i in range(K)]
        for i in range(K):
            L[i][i] += K
        r1, r2 = self.lib.combos(A, B, L)
        # B L^-T = (L^-1 B^T)^T
        ref1 = add(solve(A, B), tr(solve(L, tr(B))), -0.5)
        self.assertLess(rel(r1, ref1), 1e-12)
        self.assertLess(rel(r2, matmul(tr(A), A)), 1e-12)

    def test_cpp_source(self):
        src = self.lib.gls.cpp
        self.assertTrue(src.startswith("/* gls: matrices by reference"))
        self.assertIn('extern "C" void\ngls(', src)
        self.assertIn("ulmBLAS::", src)
        self.assertTrue(src.rstrip().endswith("}"))
        self.assertNotIn("gls_inout(", src)  # only this function
        self.assertIn(src, self.lib.source)

    def test_interface(self):
        doc = self.lib.gls_inout.__doc__
        self.assertIn("M", doc)
        self.assertIn("inout", doc)


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class WithNumPy(unittest.TestCase):

    def test_numpy_arrays(self):
        lib = Lam.lib
        X, M = np.array(Lam.X), np.array(Lam.M)
        y = np.array(Lam.y).reshape(N, 1)
        b = lib.gls(X, M, y)
        self.assertEqual(b.shape, (K, 1))
        self.assertLess(rel(b, Lam.b), 1e-12)
        ref = np.linalg.solve(X.T @ np.linalg.solve(M, X),
                              X.T @ np.linalg.solve(M, y))
        self.assertLess(np.abs(b - ref).max() / np.abs(ref).max(), 1e-12)

    def test_non_contiguous_in_is_copied(self):
        lib = Lam.lib
        Xt = np.array(Lam.X).T.copy()  # K x N, so Xt.T is N x K, F order
        b = lib.gls(Xt.T, np.array(Lam.M), np.array(Lam.y))
        self.assertLess(rel(b, Lam.b), 1e-12)

    def test_non_contiguous_inout_is_rejected(self):
        lib = Lam.lib
        Mt = np.array(Lam.M).T  # a view, not C-contiguous
        with self.assertRaisesRegex(LamError, "C-contiguous"):
            lib.gls_inout(np.array(Lam.X), Mt, np.array(Lam.y))

    def test_callbacks_get_views(self):
        lib = Lam.lib
        seen = []
        lib.bind(observe=lambda i: np.ones(N),
                 record=lambda i, b: seen.append((type(b), b.shape)))
        lib.estimate(np.array(Lam.X), np.array(Lam.M), 2)
        self.assertEqual(seen, [(np.ndarray, (K, 1))] * 2)


class Errors(unittest.TestCase):

    def test_shape_mismatch(self):
        @lam.function
        def bad(A: Matrix(3, 4), B: Matrix(3, 4)):
            return A @ B
        with self.assertRaisesRegex(LamError, "4 columns but 3 rows"):
            lam.module([bad])

    def test_loop_value_outside_loop(self):
        @lam.function
        def bad(A: Matrix(3, 3), n: Index):
            with lam.loop(n) as L:
                B = A @ A
            return B
        with self.assertRaisesRegex(LamError, "outside the loop"):
            lam.module([bad])

    def test_update_needs_inout(self):
        @lam.function
        def bad(A: Matrix(3, 3)):
            lam.update(A, A @ A)
        with self.assertRaisesRegex(LamError, "must be InOut"):
            lam.module([bad])

    def test_wrong_argument(self):
        lib = Lam.lib if hasattr(Lam, "lib") else lam.compile(gls)
        with self.assertRaisesRegex(LamError, "expected"):
            lib.gls([[1.0]], [[1.0]], [1.0])


if __name__ == "__main__":
    unittest.main()
