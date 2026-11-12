"""Vector spaces R^n and R^(m x n) in the Python frontend: sums,
differences, negation, constant and run-time scalars on either side,
division by scalars, linear combinations, with transposes, products,
solves and explicit inverses; every expression against NumPy, all in one
library (the Julia frontend's julia/UlmLam/test/vectorspace.jl in Python).
Needs NumPy.  Run:  python3 -m unittest discover -s python/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

EXPRS = [
    # vectors
    "x + y", "x - y", "-x", "x + y + z", "x - y + z", "x + y - x", "2 * x",
    "x * 2", "x / 2", "-2 * x", "2 * (x + y)", "(x + y) * 2", "(x + y) / 2",
    "2 * x + 3 * y", "2 * x - 3 * y", "(2 + 3) * x", "2 * (3 * x)",
    "-(x + y)", "-(-x)", "x - x", "0 * x", "1 * x", "a * x", "x * a",
    "x / a", "a * x + b * y", "a * (x + y)", "(a + b) * x", "a * b * x",
    "-a * x", "x - a * y", "(x + y) / a",
    # square matrices
    "A + B", "A - B", "-A", "2 * A", "A / 2", "2 * A + 3 * B",
    "a * A + b * B", "a * (A + B)", "A + B + C", "A.T + B", "(A + B).T",
    "2 * A.T", "A + 2 * eye", "A - eye", "a * eye + A", "-(A + B)",
    "A / a",
    # products in combinations
    "A @ x + y", "A @ x - y", "y - A @ x", "2 * (A @ x)", "(A @ x) * 2",
    "(A @ x) / 2", "a * A @ x + b * y", "A @ x + B @ y", "A @ (x + y)",
    "(A + B) @ x", "(A - B) @ x", "A @ B + C", "2 * A @ B - C",
    "(A + B) @ C", "A @ (B + C)", "A @ B @ x", "(A @ B) * 2", "a * (A @ B)",
    "-(A @ B)", "A @ B - B @ A", "A ** 2 @ x", "A @ eye @ x",
    # transposes, inner and outer products
    "A.T @ x", "x.T @ y", "x @ y.T", "x.T @ A @ y", "G.T @ x", "G @ G.T",
    "G.T @ G", "A.T @ A",
    # solves and inverses
    "inv(S) @ x", "inv(S) @ (x + y)", "inv(S) @ x + y", "2 * (inv(S) @ x)",
    "(inv(S) @ x) / a", "inv(S) @ x - y", "inv(2 * S) @ x",
    "inv(S + eye) @ x", "inv(a * S) @ x", "x - S @ (inv(S) @ x)",
    "inv(A) @ x", "inv(A) @ (2 * x)", "inv(A + eye) @ x", "inv(A) @ B",
    "inv(A)", "inv(S)", "inv(S) @ A + B", "inv(A.T) @ x", "inv(A) + B",
    "2 * inv(S)", "inv(S) - eye", "A ** -1", "A ** 0",
    # rectangular matrices
    "G + H", "G - H", "-G", "2 * G", "G / 2", "a * G", "G / a",
    "a * G + b * H", "a * (G + H)", "(a + b) * G", "G + H - G", "0 * G",
    "G.T + H.T", "(G + H).T", "2 * G.T", "(G + H).T @ x",
    "G.T @ x + G.T @ y", "A @ G + H", "A @ (G + H)", "(A + B) @ G",
    "G @ G.T + A", "inv(S) @ G", "inv(S) @ (G + H)", "G.T @ (inv(S) @ G)",
    "(G - H).T @ (x - y)",
]
NAMES = ("x", "y", "z", "A", "B", "C", "G", "H", "S", "a", "b", "eye")


@unittest.skipIf(np is None, "needs NumPy")
class VectorSpaces(unittest.TestCase):
    def test_against_numpy(self):
        from ulm.lam import Float, Matrix, Vector
        rng = np.random.default_rng(3)
        n, k = 6, 4
        S0 = rng.normal(size=(n, n))
        vals = {"x": rng.normal(size=(n, 1)), "y": rng.normal(size=(n, 1)),
                "z": rng.normal(size=(n, 1)), "A": rng.normal(size=(n, n)),
                "B": rng.normal(size=(n, n)), "C": rng.normal(size=(n, n)),
                "G": rng.normal(size=(n, k)), "H": rng.normal(size=(n, k)),
                "S": S0 @ S0.T + n * np.eye(n), "a": 1.7, "b": -0.6}
        types = {"x": Vector("n"), "y": Vector("n"), "z": Vector("n"),
                 "A": Matrix("n", "n"), "B": Matrix("n", "n"),
                 "C": Matrix("n", "n"), "G": Matrix("n", "k"),
                 "H": Matrix("n", "k"), "S": Matrix("n", "n", "spd"),
                 "a": Float, "b": Float}
        fns, used = [], []
        for i, e in enumerate(EXPRS):
            code = compile(e, "<expr>", "eval")
            u = [v for v in NAMES[:-1] if v in code.co_names]
            src = (f"def vs{i}({', '.join(u)}):\n"
                   f"    eye = lam.eye('n')\n    return {e}\n")
            ns = {"lam": lam, "inv": lam.inv}
            exec(src, ns)
            fn = ns[f"vs{i}"]
            fn.__annotations__ = {v: types[v] for v in u}
            fns.append(lam.function(fn))
            used.append(u)
        lib = lam.compile(fns, name="vector_spaces_py")
        for i, e in enumerate(EXPRS):
            args = [vals[v] for v in used[i]]
            # (NumPy's ** is elementwise: the matrix power by matrix_power)
            ref = eval(e.replace("A ** 2", "mpow(A, 2)")
                        .replace("A ** -1", "mpow(A, -1)")
                        .replace("A ** 0", "mpow(A, 0)"),
                       {"inv": np.linalg.inv, "eye": np.eye(n),
                        "mpow": np.linalg.matrix_power,
                        **{v: vals[v] for v in used[i]}})
            r = getattr(lib, f"vs{i}")(*args)
            with self.subTest(expr=e):
                np.testing.assert_allclose(np.ravel(r), np.ravel(ref),
                                           rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
