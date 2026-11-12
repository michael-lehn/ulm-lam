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

Sizes can be names, `Matrix("n", "m", "full_rank")`: they are run-time
parameters taken from the arrays, so one compiled library serves all sizes
(and where the best evaluation depends on the sizes, the generated code
chooses at run time).

`python/ulm/lam/` is the frontend (tracing Python to `ulm_lam` IR, loops,
external functions with Python callbacks, parameter intents, symmetric
matrices stored as one triangle: `Matrix(n, n, "spd", stored="lower")`) and
the runtime (compilation, workspaces, calling).  A factorisation that fails
numerically (a matrix assumed spd that is not, a singular one) raises
`lam.InfoError`.  Temporaries live in a 64-byte aligned workspace that the library
reuses (`lib.workspace`), or in one passed as `work=lam.Workspace(size)`.  See `python/examples/kalman.py`,
`python/examples/adaptive_kalman.py` (the covariance updated in place by a
Python callback in every step) and `python/tests/`.  NumPy is optional.

## Showcase: Gaussian process regression

`python/examples/gp.py` (`make gp`) predicts with a GP (Gaussian process)
from its textbook formula,

```python
@lam.function
def gp(K: Matrix("n", "n", "spd"), Ks: Matrix("n", "m"),
       Kss: Matrix("m", "m", "symmetric"), y: Vector("n")):
    mean = Ks.T @ K.inv @ y
    cov = Kss - Ks.T @ K.inv @ Ks
    return mean, cov
```

and ulm.lam derives what scikit-learn's `GaussianProcessRegressor` does by
hand: one Cholesky factorisation, `L^-1 K*` once (trsm), `K** - V^T V` by
syrk, no inverse.  On public data, downloaded into `build/data/`: the
Mauna Loa CO2 forecast of Rasmussen and Williams (NOAA, National Oceanic
and Atmospheric Administration; plot in `build/gp/co2.png`) and hourly bike
rentals (UCI, University of California, Irvine), timed for n up to 8000
(`--full`: 16000) against NumPy, SciPy and scikit-learn; ulm.lam runs on
ulmBLAS and, with `backend="cblas"`, on the OpenBLAS of NumPy.  Optional:
SciPy, scikit-learn, Matplotlib.  `examples/gp_cluster/` runs it with n up
to 100 000 on bwUniCluster 3.0: ulm.lam on MKL is as fast as the same steps
written by hand with MKL; ulmBLAS is 3.8x slower there (thread scaling).

## Conventions of the generated code

- Matrices by reference (row-major), owned by the caller; parameters are
  `in` (read only), `inout` (may be overwritten) or `out`.  Sizes may be
  run-time parameters (`size_t`, attribute `ulm_lam.dims`).
- Every function returns an `int64_t info`, as LAPACK routines do: 0, k > 0
  if its check k failed (a factorisation, or a call of another generated
  function), -i if argument i is illegal (a workspace that is too small).
  The interface comment in front of each function lists the checks.
- Temporaries live in a caller-provided workspace (`work`,
  `*workspace_info`: -1 query, 0 allocate, > 0 its size in bytes); a query
  accumulates those of the callees.
- ulmBLAS is called with `IndexType = std::ptrdiff_t`.
- A second lowering, `BACKEND=cblas` (`lam.compile(..., backend="cblas")`,
  or `ULM_LAM_BACKEND=cblas`): the same kernel calls on CBLAS and LAPACK
  (MKL, OpenBLAS, Accelerate) via `include/ulm_lam/cblas_backend.h`, so
  that the mapping can be measured independently of ulmBLAS.  Link flags:
  `ULM_LAM_BLAS_LIBS` (default: Homebrew's OpenBLAS).

## Document

The design document is `doc/ulm-lam.pdf`, built with LuaLaTeX from
`doc/ulm-lam.tex`.

| Path | Content |
|---|---|
| `include/ulm_lam/*.td` | ODS (Operation Definition Specification) for the `ulm.lam` (algebra) and `ulm_lamk` (kernels) dialects, interfaces, passes, kernel catalogue |
| `lib/` | the implementation, an mlir-opt plugin: both dialects, `ulm-lam-normalize`, `ulm-lam-infer-properties`, `ulm-lam-select-kernels` (strategy=gmc, with inverses), `ulm-lam-strip-properties`, `ulm-lam-results-to-out-params`, `ulm-lam-workspace`, `ulm-lam-info`, `convert-ulm-lamk-to-ulmblas` |
| `python/ulm/lam/` | the Python frontend and runtime (`from ulm import lam`) |
| `scripts/ulm-lam-compile.sh` | the pipeline: ulm_lam ops to C++ calling ulmBLAS (writes every stage) |
| `examples/gls/` | generalised least squares: hand-written design stages 1–4; from stage 2 to C++ by passes; `gls_inout.mlir` with M and y declared `inout` |
| `examples/kalman/` | formula (1) in a loop, a new measurement y_i in every step (Kalman measurement update with an uninformative prior): factorisations before the loop, O(n²) per step; `adaptive.mlir`: the covariance is updated in place in every step by an external function |
| `examples/loop/` | `X += A*B*C` in a loop: no allocation or copy inside the loop |
| `examples/compose/` | a function with a workspace calling functions with workspaces (a generated one and an external one): the workspace query accumulates |
| `examples/dynamic/` | GLS with sizes as run-time parameters: one compiled function for six sizes |
| `examples/cases/` | inverses (LU, Cholesky, triangular, from the left and the right) and sums, checked numerically |

Every example keeps the C++ generated from it in `examples/*/generated/`
(written by `make`).

What is checked (`make`):

- `make ods`: every `mlir-tblgen` backend accepts the ODS.
- `make stages`: the hand-written GLS stages 1–4 parse and verify with the
  implemented dialects; stage 3 bufferizes into exactly the copies stage 4
  shows.
- `make gls`: from the normal form (stage 2), kernel selection finds the
  derivation (two Cholesky factorisations, W = L^-1 X, S = W^T W by syrk,
  solves),
  the C++ runs and is verified against an independent LU solve.
- `make kalman`: formula (1) in a loop with changing y; everything that does
  not depend on y runs once before the loop, and if the loop runs often
  enough (decided at run time from the trip count) the whole operator
  (H^T R^-1 H)^-1 H^T R^-1 is formed there too, leaving one gemv per step;
  compared with the same code
  without loop-invariant code motion and with explicit inverses.  Second
  example: the covariance is updated in place in every step
  (`update_covariance(i, R)`, R by reference).
- `make loop`: kernel selection and memory placement of the loop example;
  the C++ runs and is verified against naive C loops.
- `make compose`: the workspace query of a function accumulates those of
  its callees; given, self-allocated and too small workspaces; a failing
  callee makes the caller return its info at once.
- `make cases`: eight kernel selection cases against naive evaluations;
  info for an indefinite and a singular matrix and a too small workspace.
- `make dynamic`: one compiled GLS with run-time sizes for six sizes.
- `make gp`: the GP showcase above (needs the network once).
- `make python`: the Python frontend: 36 unit tests (among them symmetric
  matrices stored as one triangle, with NaN in the other one) and the two
  Kalman examples against NumPy.

Requirements: Homebrew `llvm` (LLVM, originally "Low Level Virtual Machine";
MLIR 23), ulmBLAS (`ULMBLAS_DIR`, default `~/git/ulmBLAS`), TeX Live with
LuaLaTeX.

```sh
make            # ods, plugin, stages, loop, gls, kalman, compose, cases, dynamic, python, doc
```
