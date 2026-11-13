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
from ulm.lam import Float, Index, InOut, LamError, Matrix, Vector  # noqa: E402

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
        self.assertIn('extern "C" int64_t\ngls(', src)
        self.assertRegex(src, r"(ulmBLAS|ulm_lam_cblas)::")
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


class Rewrites(unittest.TestCase):

    def test_inverse_of_product_factorwise(self):
        """(R R)^-1 with R spd: R R is spd (property rule), and it is
        inverted factor by factor: one Cholesky factorisation of R, R R is
        never formed (no N x N gemm)."""

        @lam.function
        def gls_squared(X: Matrix(N, K, "full_rank"), R: Matrix(N, N, "spd"),
                        y: Vector(N)):
            M = R @ R
            return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

        lib = lam.compile(gls_squared)
        X, R, y = rand(N, K), spd(N), [rnd.uniform(-1, 1) for _ in range(N)]
        M, Y = matmul(R, R), [[v] for v in y]
        MiX, Miy = solve(M, X), solve(M, Y)
        ref = flat(solve(matmul(tr(X), MiX), matmul(tr(X), Miy)))
        self.assertLess(rel(lib.gls_squared(X, R, y), ref), 1e-10)
        src = lib.gls_squared.cpp
        self.assertEqual(src.count("potrf<std::ptrdiff_t>(true, %d," % N), 1)
        self.assertNotIn("gemm<std::ptrdiff_t>(%d, %d" % (N, N), src)
        self.assertNotIn("getrf", src)


class Info(unittest.TestCase):

    def test_not_positive_definite_raises(self):
        """M is declared spd but is not: the Cholesky factorisation fails,
        the function returns info 1, Python raises InfoError (no abort)."""
        bad = spd(N)
        for i in range(N):
            bad[i][i] = -1.0
        lib = lam.compile(gls, name="info_gls")
        with self.assertRaises(lam.InfoError) as e:
            lib.gls(rand(N, K), bad, [1.0] * N)
        self.assertEqual(e.exception.info, 1)
        self.assertIn("not positive definite", str(e.exception))

    def test_loop_stops_after_failure(self):
        """A failure in a loop: the remaining iterations do nothing."""
        calls = []
        lib = lam.compile(estimate, name="info_loop")
        bad = spd(N)
        for i in range(N):
            bad[i][i] = -1.0
        lib.bind(observe=lambda i: calls.append(i) or [1.0] * N,
                 record=lambda i, b: calls.append(("record", i)))
        with self.assertRaises(lam.InfoError):
            lib.estimate(rand(N, K), bad, 5)
        self.assertEqual(calls, [])


@unittest.skipIf(np is None, "NumPy is not installed")
class Dynamic(unittest.TestCase):
    """Sizes as names: run-time parameters, taken from the arrays."""

    @classmethod
    def setUpClass(cls):
        obs = lam.extern("observe_n", [Index], Vector("n"))
        rec = lam.extern("record_m", [Index, Vector("m")])

        @lam.function
        def gls_d(X: Matrix("n", "m", "full_rank"),
                  M: Matrix("n", "n", "spd"), y: Vector("n")):
            return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

        @lam.function
        def estimate_d(H: Matrix("n", "m", "full_rank"),
                       R: Matrix("n", "n", "spd"), steps: Index):
            with lam.loop(steps) as L:
                y = obs(L.index)
                rec(L.index, (H.T @ R.inv @ H).inv @ H.T @ R.inv @ y)

        @lam.function
        def calls_d(X: Matrix("n", "m", "full_rank"),
                    M: Matrix("n", "n", "spd"), y: Vector("n")):
            return gls_d(X, M, y)

        @lam.function
        def chain_d(A: Matrix("n", "k"), B: Matrix("k", "m"),
                    C: Matrix("m", "p")):
            return A @ B @ C

        cls.lib = lam.compile([gls_d, estimate_d, calls_d, chain_d],
                              name="dynamic")
        cls.rng = np.random.default_rng(3)

    def problem(self, n, m):
        X = self.rng.uniform(-1, 1, (n, m))
        A = self.rng.uniform(-1, 1, (n, n))
        return X, A @ A.T + n * np.eye(n), self.rng.uniform(-1, 1, n)

    def test_one_library_several_sizes(self):
        for n, m in [(5, 2), (60, 7), (200, 30)]:
            X, M, y = self.problem(n, m)
            ref = np.linalg.solve(X.T @ np.linalg.solve(M, X),
                                  X.T @ np.linalg.solve(M, y))
            self.assertLess(np.abs(self.lib.gls_d(X, M, y)[:, 0] - ref)
                            .max(), 1e-12)
            self.assertLess(np.abs(self.lib.calls_d(X, M, y)[:, 0] - ref)
                            .max(), 1e-12)
            # temporaries: L (n x n), K (m x m), W (n x m), L^-1 y (n)
            up = lambda b: (b + 63) // 64 * 64  # noqa: E731
            self.assertEqual(self.lib.gls_d.workspace_size(X, M, y),
                             up(8 * n * n) + up(8 * m * m) + up(8 * n * m)
                             + up(8 * n))

    def test_loop_with_callbacks(self):
        n, m, steps = 40, 5, 3
        X, M, _ = self.problem(n, m)
        ys = self.rng.uniform(-1, 1, (steps, n))
        out = np.empty((steps, m))

        def record(i, b):
            out[i] = b[:, 0]

        self.lib.bind(observe_n=lambda i: ys[i], record_m=record)
        self.lib.estimate_d(X, M, steps)
        ref = np.linalg.solve(X.T @ np.linalg.solve(M, X),
                              X.T @ np.linalg.solve(M, ys.T)).T
        self.assertLess(np.abs(out - ref).max(), 1e-12)

    def test_lu_and_transposed_operands(self):
        """LU with run-time sizes (crashed form-expressions before the
        operands of a call were deduplicated), solves with a transposed
        right-hand side (gecopy) and from the right with LU (getrs on the
        transposed view)."""
        @lam.function
        def lu_d(M: Matrix("n", "n"), B: Matrix("n", "k")):
            return M.inv @ B

        @lam.function
        def lu_t(M: Matrix("n", "n"), C: Matrix("k", "n")):
            return M.inv @ C.T

        @lam.function
        def right_lu(C: Matrix("k", "n"), M: Matrix("n", "n")):
            return C @ M.inv

        @lam.function
        def chol_t(M: Matrix("n", "n", "spd"), C: Matrix("k", "n")):
            return M.inv @ C.T

        lib = lam.compile([lu_d, lu_t, right_lu, chol_t], name="transposed")
        n, k = 23, 4
        A = self.rng.uniform(-1, 1, (n, n)) + n * np.eye(n)
        S = A @ A.T
        B = self.rng.uniform(-1, 1, (n, k))
        C = self.rng.uniform(-1, 1, (k, n))
        close = lambda x, r: np.abs(x - r).max() / np.abs(r).max() < 1e-12  # noqa: E731
        self.assertTrue(close(lib.lu_d(A, B), np.linalg.solve(A, B)))
        self.assertTrue(close(lib.lu_t(A, C), np.linalg.solve(A, C.T)))
        self.assertTrue(close(lib.right_lu(C, A), C @ np.linalg.inv(A)))
        self.assertTrue(close(lib.chol_t(S, C), np.linalg.solve(S, C.T)))
        self.assertIn("ulm_lam_gecopy(", lib.chol_t.cpp)

    def test_operator_hoisted_by_trip_count(self):
        """b_i = (H^T R^-1 H)^-1 H^T R^-1 y_i: with many steps the operator
        is formed before the loop (one gemv per step), with few steps the
        solves run in every step; the generated code decides from the
        trip count (n = 80, m = 60: from 8 steps on)."""
        n, m = 80, 60
        X, M, _ = self.problem(n, m)
        cpp = self.lib.estimate_d.cpp
        self.assertEqual(cpp.count("ulmBLAS::gemv")
                         + cpp.count("ulm_lam_cblas::gemv"), 2)
        for steps in (3, 20):
            ys = self.rng.uniform(-1, 1, (steps, n))
            out = np.empty((steps, m))

            def record(i, b):
                out[i] = b[:, 0]

            self.lib.bind(observe_n=lambda i: ys[i], record_m=record)
            self.lib.estimate_d(X, M, steps)
            ref = np.linalg.solve(X.T @ np.linalg.solve(M, X),
                                  X.T @ np.linalg.solve(M, ys.T)).T
            self.assertLess(np.abs(out - ref).max() / np.abs(ref).max(),
                            1e-11)

    def test_plan_chosen_at_run_time(self):
        for n, k, m, p in [(60, 3, 60, 3), (3, 60, 3, 60)]:
            A, B, C = (self.rng.uniform(-1, 1, s)
                       for s in [(n, k), (k, m), (m, p)])
            self.assertLess(np.abs(self.lib.chain_d(A, B, C) - A @ B @ C)
                            .max(), 1e-12)
            # the temporary of the cheaper plan: A B (n x m) or B C (k x p)
            left = n * k * m + n * m * p <= k * m * p + n * k * p
            tmp = n * m if left else k * p
            self.assertEqual(self.lib.chain_d.workspace_size(A, B, C),
                             (8 * tmp + 63) // 64 * 64)
        self.assertIn("::gemm<", self.lib.chain_d.cpp)
        self.assertIn("<=", self.lib.chain_d.cpp)  # the cost comparison

    def test_workspaces(self):
        """The library's workspace is reused (and grows); an explicit one
        can be given; one that is too small raises InfoError (-i)."""
        lib = self.lib
        X, M, y = self.problem(80, 6)
        ref = lib.gls_d(X, M, y)
        need = lib.gls_d.workspace_size(X, M, y)
        self.assertGreaterEqual(lib.workspace.size, need)
        self.assertEqual(lib.workspace.address % 64, 0)
        address = lib.workspace.address
        lib.gls_d(*self.problem(20, 3))  # smaller: the same memory
        self.assertEqual(lib.workspace.address, address)
        ws = lam.Workspace(need)
        self.assertTrue(np.array_equal(lib.gls_d(X, M, y, work=ws), ref))
        arr = np.zeros(need, dtype=np.int8)
        self.assertTrue(np.array_equal(lib.gls_d(X, M, y, work=arr), ref))
        with self.assertRaises(lam.InfoError) as e:
            lib.gls_d(X, M, y, work=lam.Workspace(need - 64))
        self.assertLess(e.exception.info, 0)
        self.assertIn("workspace too small", str(e.exception))

    def test_inconsistent_sizes(self):
        X, M, y = self.problem(6, 2)
        with self.assertRaises(LamError):
            self.lib.gls_d(X, M[:5, :5], y)

    def test_interface_names_the_sizes(self):
        doc = self.lib.gls_d.__doc__
        self.assertIn("sizes n, m", doc)
        self.assertIn("double[n][m]", doc)


class Cache(unittest.TestCase):

    def test_interrupted_build_is_not_reused(self):
        """A changed program whose build did not finish (only its .mlir was
        written) must be compiled, not served from the old library."""
        import tempfile

        @lam.function
        def square(A: Matrix(4, 4)):
            return A @ A

        @lam.function
        def cube(A: Matrix(4, 4)):
            return A @ A @ A

        with tempfile.TemporaryDirectory() as d:
            lam.compile(square, name="cached", workdir=d)
            # the state an interrupted build of `cube` leaves behind
            Path(d, "cached.mlir").write_text(lam.frontend.module([cube]))
            lib = lam.compile(cube, name="cached", workdir=d)
            A = [[2.0 * (i == j) for j in range(4)] for i in range(4)]
            self.assertAlmostEqual(lib.cube(A)[0][0], 8.0)


@unittest.skipIf(np is None, "NumPy is not installed")
class Storage(unittest.TestCase):
    """Symmetric matrices stored as one triangle (stored="lower"/"upper"):
    the other triangle holds NaN, so any kernel that reads it shows."""

    n, k = 30, 7

    @classmethod
    def setUpClass(cls):
        n, k = cls.n, cls.k
        lo = dict(stored="lower")
        up = dict(stored="upper")

        @lam.function
        def left(A: Matrix(n, n, "symmetric", **lo), B: Matrix(n, k)):
            return A @ B

        @lam.function
        def right(A: Matrix(n, n, "symmetric", **up), C: Matrix(k, n)):
            return C @ A

        @lam.function
        def left_t(A: Matrix(n, n, "symmetric", **lo), C: Matrix(k, n)):
            return A @ C.T

        @lam.function
        def vec(A: Matrix(n, n, "symmetric", **up), x: Vector(n)):
            return A @ x

        @lam.function
        def both(A: Matrix(n, n, "symmetric", **lo),
                 S: Matrix(n, n, "symmetric", **up)):
            return A @ S

        @lam.function
        def chol_up(R: Matrix(n, n, "spd", **up), B: Matrix(n, k)):
            return R.inv @ B

        @lam.function
        def chol_lo(R: Matrix(n, n, "spd", **lo), B: Matrix(n, k)):
            return R.inv @ B

        @lam.function
        def plus(A: Matrix(n, n, "symmetric", **lo), G: Matrix(n, n)):
            return A + G

        @lam.function
        def lu_sym(A: Matrix(n, n, "symmetric", **up), B: Matrix(n, k)):
            return A.inv @ B

        cls.lib = lam.compile([left, right, left_t, vec, both, chol_up,
                               chol_lo, plus, lu_sym], name="storage")
        rng = np.random.default_rng(1)
        cls.rng = rng

    def sym(self, spd=False):
        n = self.n
        M = self.rng.uniform(-1, 1, (n, n))
        M = M + M.T
        return M @ M.T + n * np.eye(n) if spd else M

    def half(self, M, tri):
        H = M.copy()
        H[np.triu_indices(self.n, 1) if tri == "lower"
          else np.tril_indices(self.n, -1)] = np.nan
        return H

    def check(self, got, ref):
        got = np.asarray(got)
        self.assertFalse(np.isnan(got).any(), "the NaN triangle was read")
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-12)

    def test_products(self):
        n, k, lib = self.n, self.k, self.lib
        A, S = self.sym(), self.sym()
        B = self.rng.uniform(-1, 1, (n, k))
        C = self.rng.uniform(-1, 1, (k, n))
        x = self.rng.uniform(-1, 1, (n, 1))
        self.check(lib.left(self.half(A, "lower"), B), A @ B)
        self.check(lib.right(self.half(A, "upper"), C), C @ A)
        self.check(lib.left_t(self.half(A, "lower"), C), A @ C.T)
        self.check(lib.vec(self.half(A, "upper"), x), A @ x)
        self.check(lib.both(self.half(A, "lower"), self.half(S, "upper")),
                   A @ S)
        for f in ("left", "right", "left_t", "vec"):
            self.assertNotIn("gemm", getattr(lib, f).cpp)
            self.assertNotIn("gemv", getattr(lib, f).cpp)
            self.assertNotIn("sycomplete(", getattr(lib, f).cpp)
        self.assertEqual(lib.both.cpp.count("ulm_lam_sycomplete("), 1)

    def test_factorisations_and_sums(self):
        n, k, lib = self.n, self.k, self.lib
        A, R = self.sym(), self.sym(spd=True)
        B = self.rng.uniform(-1, 1, (n, k))
        G = self.rng.uniform(-1, 1, (n, n))
        self.check(lib.chol_up(self.half(R, "upper"), B),
                   np.linalg.solve(R, B))
        self.check(lib.chol_lo(self.half(R, "lower"), B),
                   np.linalg.solve(R, B))
        self.assertIn("potrf<std::ptrdiff_t>(false,", lib.chol_up.cpp)
        self.check(lib.plus(self.half(A, "lower"), G), A + G)
        self.check(lib.lu_sym(self.half(A, "upper"), B),
                   np.linalg.solve(A, B))

    def test_interface_names_the_triangle(self):
        self.assertIn("lower triangle", self.lib.left.cpp)

    def test_syrk_results(self):
        """X^T X is computed by syrk (one triangle); the result is completed
        where the whole matrix is needed (returned, summed), used by symm
        in a product and by potrf directly."""
        n, k = self.n, self.k

        @lam.function
        def gram(X: Matrix(n, k)):
            return X.T @ X

        @lam.function
        def gram_use(X: Matrix(n, k, "full_rank"), Y: Matrix(k, 5)):
            S = X.T @ X
            return S @ Y, S.inv @ Y, S + S

        lib = lam.compile([gram, gram_use], name="syrk")
        X = self.rng.uniform(-1, 1, (n, k))
        Y = self.rng.uniform(-1, 1, (k, 5))
        S = X.T @ X
        self.check(lib.gram(X), S)
        a, b, c = lib.gram_use(X, Y)
        self.check(a, S @ Y)
        self.check(b, np.linalg.solve(S, Y))
        self.check(c, 2 * S)
        self.assertIn("sylrk", lib.gram.cpp)
        self.assertNotIn("gemm", lib.gram.cpp)
        self.assertIn("sylmm", lib.gram_use.cpp)

    def test_syrk_update(self):
        """C - V^T V (the GP covariance K** - V^T V): syrk updates the
        lower triangle of a symmetric C (stored completely or as the lower
        triangle); a general C needs gemm."""
        n, k = self.n, self.k

        @lam.function
        def upd(C: Matrix(k, k, "symmetric"), V: Matrix(n, k)):
            return C - V.T @ V

        @lam.function
        def upd_lo(C: Matrix(k, k, "symmetric", stored="lower"),
                   V: Matrix(n, k)):
            return 2 * C - 3 * (V.T @ V)

        @lam.function
        def upd_general(C: Matrix(k, k), V: Matrix(n, k)):
            return C - V.T @ V

        lib = lam.compile([upd, upd_lo, upd_general], name="syrk_update")
        V = self.rng.uniform(-1, 1, (n, k))
        M = self.rng.uniform(-1, 1, (k, k))
        C = M + M.T
        H = C.copy()
        H[np.triu_indices(k, 1)] = np.nan
        self.check(lib.upd(C, V), C - V.T @ V)
        self.check(lib.upd_lo(H, V), 2 * C - 3 * V.T @ V)
        self.check(lib.upd_general(M, V), M - V.T @ V)
        for f in ("upd", "upd_lo"):
            self.assertIn("sylrk", getattr(lib, f).cpp)
            self.assertNotIn("gemm", getattr(lib, f).cpp)
        self.assertIn("gemm", lib.upd_general.cpp)

    def test_stored_needs_symmetric(self):
        with self.assertRaises(LamError):
            Matrix(4, 4, stored="lower")


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


@unittest.skipIf(np is None, "NumPy is not installed")
class Selection(unittest.TestCase):
    """Plans found by comparing the selection with an e-graph
    (experiments/egraph/lamp_fuzz.py): a product with a sum as a factor is
    distributed where that is cheaper, a Gram matrix inside a longer product
    is a syrk, and an inverse used as a value is formed by potri / getri."""

    @classmethod
    def setUpClass(cls):
        @lam.function
        def lora(W: Matrix("n", "n"), B: Matrix("n", "r"),
                 A: Matrix("r", "n"), x: Vector("n")):
            return (W + B @ A) @ x

        @lam.function
        def solve_in_sum(C: Matrix(N, N), B: Matrix(N, N, "non_singular"),
                         D: Matrix(N, K), u: Vector(K)):
            return (C + B.T.inv) @ D @ u

        @lam.function
        def gram_left(A: Matrix(K, N), B: Matrix(K, K)):
            return A @ A.T @ B

        @lam.function
        def inverses(S: Matrix(N, N, "spd"), A: Matrix(N, N, "non_singular"),
                     C: Matrix(N, N)):
            return S.inv, A.inv, C + A.inv.T

        @lam.function
        def information(P: Matrix(N, N, "spd"), H: Matrix(K, N),
                        R: Matrix(K, K, "spd")):
            return (P.inv + H.T @ R.inv @ H).inv

        cls.lib = lam.compile([lora, solve_in_sum, gram_left, inverses,
                               information], name="selection")
        cls.r = np.random.default_rng(3)

    def spd(self, n):
        a = self.r.standard_normal((n, n))
        return a @ a.T + n * np.eye(n)

    def test_lora_distributed(self):
        for n, r in ((60, 4), (8, 30)):
            W, B = self.r.standard_normal((n, n)), self.r.standard_normal((n, r))
            A, x = self.r.standard_normal((r, n)), self.r.standard_normal(n)
            ref = (W + B @ A) @ x
            got = self.lib.lora(W, B, A, x).ravel()
            self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-12)
        # W x + B (A x): no n x n product is formed
        self.assertNotIn("gemm", self.lib.lora.cpp)

    def test_solve_in_sum_needs_no_inverse(self):
        C, B = self.r.standard_normal((N, N)), self.spd(N)
        D, u = self.r.standard_normal((N, K)), self.r.standard_normal(K)
        ref = (C + np.linalg.inv(B.T)) @ D @ u
        got = self.lib.solve_in_sum(C, B, D, u).ravel()
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-10)
        self.assertNotIn("getri", self.lib.solve_in_sum.cpp)

    def test_gram_inside_product(self):
        A, B = self.r.standard_normal((K, N)), self.r.standard_normal((K, K))
        ref = A @ A.T @ B
        got = self.lib.gram_left(A, B)
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-12)
        src = self.lib.gram_left.cpp
        self.assertRegex(src, r"syrk|sylrk")   # one triangle
        self.assertRegex(src, r"symm|sylmm")   # read by symm

    def test_explicit_inverses(self):
        S, A, C = self.spd(N), self.r.standard_normal((N, N)), \
            self.r.standard_normal((N, N))
        si, ai, s = self.lib.inverses(S, A, C)
        for got, ref in ((si, np.linalg.inv(S)), (ai, np.linalg.inv(A)),
                         (s, C + np.linalg.inv(A).T)):
            self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(),
                            1e-9)
        src = self.lib.inverses.cpp
        self.assertIn("potri", src)
        self.assertIn("getri", src)

    def test_information_form(self):
        P, H, R = self.spd(N), self.r.standard_normal((K, N)), self.spd(K)
        ref = np.linalg.inv(np.linalg.inv(P) + H.T @ np.linalg.solve(R, H))
        got = self.lib.information(P, H, R)
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-10)


@unittest.skipIf(np is None, "NumPy is not installed")
class Identity(unittest.TestCase):
    """alpha * I in sums (lam.eye), alpha a constant or a Float parameter
    known at run time: added to the diagonal (adddiag), never formed."""

    @classmethod
    def setUpClass(cls):
        n, m = 30, 4

        @lam.function
        def noise(K: Matrix(n, n, "symmetric"), s2: Float, B: Matrix(n, m)):
            return lam.assume(K + s2 * lam.eye(n), "spd").inv @ B

        @lam.function
        def noise_d(K: Matrix("n", "n", "symmetric"), s2: Float,
                    B: Matrix("n", "m")):
            return lam.assume(K + s2 * lam.eye("n"), "spd").inv @ B

        @lam.function
        def terms(A: Matrix(n, n), B: Matrix(n, n), s: Float):
            return A @ B + 2 * lam.eye(n) - s * lam.eye(n) + A

        @lam.function
        def gram(X: Matrix(n, m), s: Float):
            # syrk computes one triangle; adddiag keeps it
            return X.T @ X + s * lam.eye(m)

        cls.lib = lam.compile([noise, noise_d, terms, gram], name="identity")
        cls.rng = np.random.default_rng(5)
        cls.n, cls.m = n, m

    def kernel(self, n):
        x = self.rng.uniform(-1, 1, (n, 2))
        return np.exp(-((x[:, None] - x[None]) ** 2).sum(-1))

    def test_noise(self):
        n, m = self.n, self.m
        K, B = self.kernel(n), self.rng.uniform(-1, 1, (n, m))
        ref = np.linalg.solve(K + 0.1 * np.eye(n), B)
        for f in (self.lib.noise, self.lib.noise_d):
            got = f(K, 0.1, B)
            self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(),
                            1e-10)
            self.assertIn("ulm_lam_adddiag(", f.cpp)
        got = self.lib.noise_d(self.kernel(50), 0.5,
                               self.rng.uniform(-1, 1, (50, 3)))
        self.assertEqual(got.shape, (50, 3))

    def test_wrong_assumption_raises(self):
        K, B = self.kernel(self.n), np.ones((self.n, self.m))
        with self.assertRaises(lam.InfoError) as e:
            self.lib.noise(K, -3.0, B)
        self.assertIn("computed from", str(e.exception))

    def test_several_terms(self):
        n = self.n
        A, B = self.rng.uniform(-1, 1, (n, n)), self.rng.uniform(-1, 1, (n, n))
        ref = A @ B + (2 - 0.7) * np.eye(n) + A
        got = self.lib.terms(A, B, 0.7)
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-13)

    def test_one_triangle_kept(self):
        X = self.rng.uniform(-1, 1, (self.n, self.m))
        got = self.lib.gram(X, 0.25)
        ref = X.T @ X + 0.25 * np.eye(self.m)
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-13)
        self.assertIn("sylrk", self.lib.gram.cpp)

    def test_runtime_scalars_on_products(self):
        """Bayesian linear regression: beta * (Phi^T Phi), beta * A^-1 ...,
        1 / beta * I; scalar arithmetic of Float parameters."""
        @lam.function
        def blr(Phi: Matrix("n", "p"), t: Vector("n"), Phis: Matrix("m", "p"),
                alpha: Float, beta: Float):
            A = lam.assume(alpha * lam.eye("p") + beta * (Phi.T @ Phi), "spd")
            w = beta * A.inv @ Phi.T @ t
            return Phis @ w, Phis @ A.inv @ Phis.T + (1 / beta) * lam.eye("m")

        @lam.function
        def scaled(A: Matrix("n", "n"), B: Matrix("n", "n"), s: Float):
            return (s - 1) * A @ B + s * A + B / s

        lib = lam.compile([blr, scaled], name="scalars")
        rng = self.rng
        n, p, m = 50, 8, 6
        Phi, Phis, t = (rng.normal(size=(n, p)), rng.normal(size=(m, p)),
                        rng.normal(size=n))
        A = 2.0 * np.eye(p) + 25.0 * Phi.T @ Phi
        w = 25.0 * np.linalg.solve(A, Phi.T @ t)
        mean, cov = lib.blr(Phi, t, Phis, 2.0, 25.0)
        self.assertLess(np.abs(mean[:, 0] - Phis @ w).max(), 1e-12)
        ref = Phis @ np.linalg.solve(A, Phis.T) + np.eye(m) / 25.0
        self.assertLess(np.abs(cov - ref).max(), 1e-12)
        X, Y = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
        ref = 2.5 * X @ Y + 3.5 * X + Y / 3.5
        self.assertLess(np.abs(lib.scaled(X, Y, 3.5) - ref).max(), 1e-12)

    def test_scaled_product_in_sum_is_one_gemm(self):
        """W - s * (X^T R): one gemm with alpha = -s, beta = 1 into W (the
        SGD update of torch_linear.py), no gescal or geaxpy."""
        @lam.function
        def sgd(W: Matrix("p", "k"), X: Matrix("B", "p"), R: Matrix("B", "k"),
                s: Float):
            return W - s * (X.T @ R)

        lib = lam.compile(sgd, name="sgd_step")
        rng = self.rng
        W, X, R = (rng.normal(size=(7, 3)), rng.normal(size=(11, 7)),
                   rng.normal(size=(11, 3)))
        self.assertLess(np.abs(lib.sgd(W, X, R, 0.3)
                               - (W - 0.3 * X.T @ R)).max(), 1e-13)
        cpp = lib.sgd.cpp
        self.assertEqual(cpp.count("::gemm<"), 1)
        self.assertNotIn("gescal", cpp)
        self.assertNotIn("geaxpy", cpp)

    def test_float_argument(self):
        with self.assertRaises(LamError):
            self.lib.terms(np.eye(self.n), np.eye(self.n), "x")


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
