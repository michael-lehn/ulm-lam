"""foo calls bar, and bar needs a workspace (python3 examples/calls/calls.py;
the same in Julia: calls.jl)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ulm import lam  # noqa: E402
from ulm.lam import Matrix, Vector  # noqa: E402


@lam.function
def bar(M: Matrix("n", "n", "spd"), y: Vector("n")):
    return M.inv @ y


@lam.function
def foo(M: Matrix("n", "n", "spd"), N: Matrix("n", "n", "spd"),
        y: Vector("n")):
    return bar(M, bar(N, y))


rng = np.random.default_rng(1)
n = 5
A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
M, N, y = A @ A.T + n * np.eye(n), B @ B.T + n * np.eye(n), rng.normal(size=n)

lib = lam.compile(foo, name="calls_py")      # foo and bar
x = lib.foo(M, N, y)
ref = np.linalg.solve(M, np.linalg.solve(N, y))
print("error:", np.linalg.norm(x.ravel() - ref) / np.linalg.norm(ref))
print("workspace of bar:", lib.bar.workspace_size(M, y), "bytes, of foo:",
      lib.foo.workspace_size(M, N, y), "bytes")
here = Path(__file__).resolve().parent / "generated"
(here / "calls_py.mlir").write_text(lam.module([foo]))
(here / "calls_py.cpp").write_text(lib.source)
print(lib.foo.code)
print(lib.bar.code)

# The same with bar allowed to overwrite M (InOut): bar needs no workspace;
# foo2, whose M and N are only read, passes copies (in its own workspace);
# foo3, which may overwrite M and N itself, passes them on.
from ulm.lam import InOut  # noqa: E402


@lam.function
def bar2(M: InOut(Matrix("n", "n", "spd")), y: Vector("n")):
    return M.inv @ y


@lam.function
def foo2(M: Matrix("n", "n", "spd"), N: Matrix("n", "n", "spd"),
         y: Vector("n")):
    return bar2(M, bar2(N, y))


@lam.function
def foo3(M: InOut(Matrix("n", "n", "spd")), N: InOut(Matrix("n", "n", "spd")),
         y: Vector("n")):
    return bar2(M, bar2(N, y))


lib2 = lam.compile([foo2, foo3], name="calls_inout_py")
M2, N2, M3, N3 = M.copy(), N.copy(), M.copy(), N.copy()
x2, x3 = lib2.foo2(M2, N2, y), lib2.foo3(M3, N3, y)
print("errors:", np.linalg.norm(x2.ravel() - ref) / np.linalg.norm(ref), ",",
      np.linalg.norm(x3.ravel() - ref) / np.linalg.norm(ref),
      "; M after foo2 unchanged:", bool((M2 == M).all()),
      ", after foo3 overwritten:", bool((M3 != M).any()))
print("workspace of bar2:", lib2.bar2.workspace_size(M, y), "bytes, of foo2:",
      lib2.foo2.workspace_size(M, N, y), "bytes, of foo3:",
      lib2.foo3.workspace_size(M, N, y), "bytes")
(here / "calls_inout_py.mlir").write_text(lam.module([foo2, foo3]))
(here / "calls_inout_py.cpp").write_text(lib2.source)
print(lib2.foo2.code)
print(lib2.foo3.code)
print(lib2.bar2.code)

# LAPACK style: bar_ and foo_ overwrite y with the solution (and M, N with
# their Cholesky factors), so neither needs a workspace; foo_matlab is the
# user-friendly wrapper that leaves its arguments alone: the copies, and so
# the workspace, are in it only.


@lam.function
def bar_(M: InOut(Matrix("n", "n", "spd")), y: InOut(Vector("n"))):
    lam.update(y, M.inv @ y)


@lam.function
def foo_(M: InOut(Matrix("n", "n", "spd")), N: InOut(Matrix("n", "n", "spd")),
         y: InOut(Vector("n"))):
    lam.update(y, bar_(M, bar_(N, y)))


@lam.function
def foo_matlab(M: Matrix("n", "n", "spd"), N: Matrix("n", "n", "spd"),
               y: Vector("n")):
    return foo_(M, N, y)


lib3 = lam.compile([foo_matlab], name="calls_lapack_py")   # and foo_, bar_
M4, N4, y4 = M.copy(), N.copy(), y.copy()
lib3.foo_(M4, N4, y4)                 # y4 is the solution now
x5 = lib3.foo_matlab(M, N, y)
print("errors:", np.linalg.norm(y4 - ref) / np.linalg.norm(ref), ",",
      np.linalg.norm(x5.ravel() - ref) / np.linalg.norm(ref),
      "; after foo_: lower triangle of M ≈ cholesky(M):",
      bool(np.allclose(np.tril(M4), np.linalg.cholesky(M))))
print("workspace (LAPACK style) of bar:", lib3.bar_.workspace_size(M, y),
      "bytes, of foo:", lib3.foo_.workspace_size(M, N, y),
      "bytes; of foo_matlab:", lib3.foo_matlab.workspace_size(M, N, y),
      "bytes")
(here / "calls_lapack_py.mlir").write_text(lam.module([foo_matlab]))
(here / "calls_lapack_py.cpp").write_text(lib3.source)
print(lib3.foo_.code)
print(lib3.bar_.code)
print(lib3.foo_matlab.code)
