# ulm.lam: Ulm Linear Algebra Mapping

A solution of the Linear Algebra Mapping Problem (LAMP) on MLIR.

A Linnea-style compiler built on MLIR (Multi-Level Intermediate
Representation). It maps linear algebra expressions such as
`b = (X^T M^-1 X)^-1 X^T M^-1 y` onto calls to the C++ interface of
ulmBLAS: BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra
PACKage) routines as header-only templates on strided matrices.

## Python

```python
from ulm import lam
from ulm.lam import Matrix, Vector

@lam.function
def gls(X: Matrix(2000, 200, "full_rank"), M: Matrix(2000, 2000, "spd"),
        y: Vector(2000)):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

lib = lam.compile(gls)          # ulm_lam IR -> C++ with ulmBLAS -> shared library
b = lib.gls(X, M, y)            # NumPy arrays (or any float64 buffer) in and out
print(lib.gls.cpp)               # the generated C++ of gls
```

`python/ulm/lam/` is the frontend (tracing Python to `ulm_lam` IR, loops,
external functions with Python callbacks, parameter intents) and the runtime
(compilation, workspaces, calling).  See `python/examples/kalman.py`,
`python/examples/adaptive_kalman.py` (the covariance updated in place by a
Python callback in every step) and `python/tests/`.  NumPy is optional.

## Document

The design document is `doc/ulm-lam.pdf`, built with LuaLaTeX from
`doc/ulm-lam.tex`.

| Path | Content |
|---|---|
| `include/ulm_lam/*.td` | ODS (Operation Definition Specification) for the `ulm.lam` (algebra) and `ulm_lamk` (kernels) dialects, interfaces, passes, kernel catalogue |
| `lib/` | the implementation, an mlir-opt plugin: both dialects, `ulm-lam-normalize`, `ulm-lam-infer-properties`, `ulm-lam-select-kernels` (strategy=gmc, with inverses), `ulm-lam-strip-properties`, `ulm-lam-results-to-out-params`, `convert-ulm-lamk-to-ulmblas` |
| `python/ulm/lam/` | the Python frontend and runtime (`from ulm import lam`) |
| `scripts/ulm-lam-compile.sh` | the pipeline: ulm_lam ops to C++ calling ulmBLAS (writes every stage) |
| `examples/gls/` | generalised least squares: hand-written design stages 1–4; from stage 2 to C++ by passes; `gls_inout.mlir` with M and y declared `inout` |
| `examples/kalman/` | formula (1) in a loop, a new measurement y_i in every step (Kalman measurement update with an uninformative prior): factorisations before the loop, O(n²) per step; `adaptive.mlir`: the covariance is updated in place in every step by an external function |
| `examples/loop/` | `X += A*B*C` in a loop: no allocation or copy inside the loop |
| `examples/compose/` | a function with a workspace calling functions with workspaces (a generated one and an external one): the workspace query accumulates |
| `examples/cases/` | inverses (LU, Cholesky, triangular, from the left and the right) and sums, checked numerically |

Every example keeps the C++ generated from it in `examples/*/generated/`
(written by `make`).

What is checked (`make`):

- `make ods`: every `mlir-tblgen` backend accepts the ODS.
- `make stages`: the hand-written GLS stages 1–4 parse and verify with the
  implemented dialects; stage 3 bufferizes into exactly the copies stage 4
  shows.
- `make gls`: from the normal form (stage 2), kernel selection finds the
  derivation (two Cholesky factorisations, W = L^-1 X, S = W^T W, solves),
  the C++ runs and is verified against an independent LU solve.
- `make kalman`: formula (1) in a loop with changing y; everything that does
  not depend on y runs once before the loop; compared with the same code
  without loop-invariant code motion and with explicit inverses.  Second
  example: the covariance is updated in place in every step
  (`update_covariance(i, R)`, R by reference).
- `make loop`: kernel selection and memory placement of the loop example;
  the C++ runs and is verified against naive C loops.
- `make compose`: the workspace query of a function accumulates those of
  its callees; given, self-allocated and too small workspaces.
- `make python`: the Python frontend: 19 unit tests and the two Kalman
  examples against NumPy.
- `make cases`: eight kernel selection cases against naive evaluations.

Requirements: Homebrew `llvm` (LLVM, originally "Low Level Virtual Machine";
MLIR 23), ulmBLAS (`ULMBLAS_DIR`, default `~/git/ulmBLAS`), TeX Live with
LuaLaTeX.

```sh
make            # ods, plugin, stages, loop, gls, kalman, compose, cases, python, doc
```
