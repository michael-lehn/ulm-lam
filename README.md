# ulm.lam: write the formula, get the BLAS/LAPACK calls an expert would write

ulm.lam (Ulm Linear Algebra Mapping) is a compiler built on MLIR
(Multi-Level Intermediate Representation).  You write linear algebra in
Julia or Python the way it is written in a paper,
`inv(X' * inv(M) * X) * X' * inv(M) * y`, and declare what you know about
the matrices (symmetric positive definite, full rank, triangular, …).
ulm.lam derives the sequence of BLAS (Basic Linear Algebra Subprograms) and
LAPACK (Linear Algebra PACKage) calls a numerical analyst would write by
hand:

- factorisations and triangular solves instead of inverses, each
  factorisation computed once and reused;
- symmetry exploited (`syrk` instead of `gemm`: one triangle, half the
  flops);
- the cheaper order of a product, chosen at run time when the sizes are
  only known then;
- temporaries in one workspace that is reused across calls, loop state
  updated in place: no allocations inside loops;
- failed factorisations reported like LAPACK's `info`, with a description.

`@code_lam` shows what it derived.  The generated C++ calls the BLAS and
LAPACK Julia uses (OpenBLAS, or whatever libblastrampoline forwards to,
e.g. [UlmBLAS.jl](https://github.com/michael-lehn/UlmBLAS.jl), `Pkg.add("UlmBLAS")`), the
header-only C++ library [ulmBLAS](https://github.com/michael-lehn/ulmBLAS),
or any CBLAS (C interface to the BLAS) and LAPACK, such as MKL (Intel Math
Kernel Library) or Apple's Accelerate.

An early research prototype: dense real matrices, in double or single
precision (from Julia a function called with `Float32` arrays is compiled
for `Float32`; from Python `lam.compile(f, dtype="float32")`).

## Contents

- [Julia](#julia)
- [Python](#python)
- [What you can write](#what-you-can-write)
- [Measurements](#measurements)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Development](#development)
- [What is open](#what-is-open)
- [Background](#background)
- [Open questions](#open-questions)

## Julia

```julia
using UlmLam, LinearAlgebra

@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

β = gls(X, M, y)          # Julia arrays; the first call compiles (about 0.5 s)
```

```
julia> @code_lam gls(X, M, y)
gls(X::n×m full_rank, M::n×n spd, y::n)
  potrf    L1 ← chol(M)                      M = L1 L1ᵀ, Cholesky
  copy     T1 ← X
  trsm     T1 ← L1⁻¹ T1
  syrk     T2 ← T1ᵀ T1                       one triangle only
  potrf    L2 ← chol(T2)                     T2 = L2 L2ᵀ, Cholesky
  trsv     T3 ← L1⁻¹ y
  gemv     T4 ← T1ᵀ T3
  trsv     T4 ← L2⁻¹ T4
  trsv     T4 ← L2⁻ᵀ T4
  return T4
if a factorisation fails, the call throws InfoError with
  info 1:  Cholesky factorisation of M: not positive definite
  info 2:  Cholesky factorisation of a computed matrix: not positive definite
```

`inv(M)` appears twice and is factorised once; $X^T M^{-1} X$ becomes
$(L_1^{-1}X)^T(L_1^{-1}X)$, a triangular solve and a symmetric rank-k
update; no inverse is formed, and neither `M` nor `X` is overwritten.

A loop with its state carried in place and callbacks into Julia: the
Kalman filter as in the textbook,

```julia
measure = extern(:measure, [Index], Vec(:m))      # implemented in Julia
estimate = extern(:estimate, [Index, Vec(:n)])

@lam function kalman(F::Mat(:n, :n), Q::Mat(:n, :n, :spsd), H::Mat(:m, :n),
                     R::Mat(:m, :m, :spd), x0::Vec(:n), P0::Mat(:n, :n, :spd),
                     steps::Index)
    loop(steps, x0, P0) do i, x, P
        P = assume(P, :symmetric)
        x = F * x
        P = F * P * F' + Q
        S = assume(H * P * H' + R, :spd)
        K = P * H' * inv(S)
        x = x + K * (measure(i) - H * x)
        P = P - K * H * P
        estimate(i, x)
        (x, P)
    end
end

bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> ...)
x, P = kalman(F, Q, H, R, x0, P0, steps)
```

`@code_lam kalman(...)` shows one Cholesky factorisation of $S$ per step,
$K$ from two triangular solves instead of $S^{-1}$, $H P$ computed once and
reused for $S$ and for $K H P$ (evaluated as $K (H P)$), and no allocation
in the loop.

**Try it in the browser: [▶ the notebook, live on Binder](https://mybinder.org/v2/gh/michael-lehn/ulm-lam/master?urlpath=pluto%2Fopen%3Fpath%3D%2Fhome%2Fjovyan%2Fnotebook%2Fulm-lam.jl)**
([static version](https://michael-lehn.github.io/ulm-lam/)).

The Pluto notebook [`notebook/ulm-lam.jl`](notebook/ulm-lam.jl) has these
and more to play with: GP (Gaussian process) regression compared with
AbstractGPs.jl, linear attention with the order of $Q K^T V$ chosen at run
time, and the MLIR the Julia code is traced into.

## Python

```python
from ulm import lam
from ulm.lam import Matrix, Vector

@lam.function
def gls(X: Matrix("n", "m", "full_rank"), M: Matrix("n", "n", "spd"),
        y: Vector("n")):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

lib = lam.compile(gls)     # one shared library for all sizes
b = lib.gls(X, M, y)       # NumPy arrays (or any float64 buffer), no copies
print(lib.gls.code)        # the derivation, as @code_lam shows it in Julia
print(lib.gls.cpp)         # the generated C++
```

## What you can write

| | Julia | Python |
|:--|:--|:--|
| matrix | `Mat(rows, cols, props...)` | `Matrix(rows, cols, *props)` |
| vector | `Vec(n)` | `Vector(n)` |
| sizes | numbers, or symbols for run-time sizes (`:n`) | numbers, or strings (`"n"`) |
| run-time scalar | `Float` | `Float` |
| integer | `Index` (trip counts, callback arguments) | `Index` |
| product, sum | `*`, `+`, `-`, scalars | `@`, `+`, `-`, scalars |
| transpose | `X'`, `transpose(X)` | `X.T` |
| inverse, solve | `inv(A)`, `A \ B`, `B / A` | `A.inv`, `lam.inv(A)` |
| least squares | `X \ y` for a rectangular `X`, `lstsq(X, y)` | `lam.lstsq(X, y)` |
| identity | `I`, `eye(n)` (never formed) | `lam.eye(n)` |
| facts | `assume(A, :spd)` | `lam.assume(A, "spd")` |
| loops | `loop(n, carry...) do i, x... end` | `with lam.loop(n, carry=[...]) as L:` |
| callbacks | `extern(name, argtypes, result)`, `bind!` | `lam.extern(...)`, `lib.bind(...)` |
| overwritable parameter | `InOut(Mat(...))` | `InOut(Matrix(...))` |
| several results | a tuple | a tuple |
| inspect | `@code_lam f(args...)` | `print(lib.f.code)`, `lib.f.cpp` |

All vector-space operations on $\mathbb{R}^n$ and $\mathbb{R}^{m \times n}$
work, also combined with products, transposes and solves: sums,
differences, negation, constant and run-time scalars on either side,
division by scalars, linear combinations; `julia/UlmLam/test/vectorspace.jl`
checks 137 such expressions against Julia.  An inverse used as a value
(`inv(A)` as a result, `inv(S) - I`) is formed by a factorisation and a
solve with the identity; inside products it is never formed.  A scalar
such as `x' * y` comes back as a vector of length 1.

Properties: `symmetric`, `spd` (symmetric positive definite), `spsd`
(symmetric positive semidefinite), `lower_tri`, `upper_tri`, `unit_diag`,
`diagonal`, `full_rank`, `non_singular`, `orthogonal`, `permutation`,
`identity`, `zero`.  Properties of results are inferred ($X^T M^{-1} X$ is
spd when $M$ is spd and $X$ has full rank); `assume` adds what cannot be
derived, and a factorisation that finds it wrong says so (`InfoError`).

## Measurements

Apple M5, Julia 1.13 with its OpenBLAS (4 threads), from the notebook;
time and memory allocated per call:

| | ulm.lam | Julia, the formula as written | Julia, written by an expert |
|:--|--:|--:|--:|
| GLS (generalised least squares), n = 2000, m = 50 | 28 ms, 3.4 KiB | 228 ms (8.1×), 64 MiB | 26 ms, 34 MiB |
| GP regression, 4000 training and 1000 test points | 289 ms, 328 MiB | 1965 ms (6.8×), 737 MiB | AbstractGPs.jl: 331 ms, 618 MiB |
| Kalman filter, n = 1000, m = 100, 20 steps | 334 ms, 7.7 MiB | 482 ms (1.4×), 839 MiB | 390 ms, 17 MiB |
| linear attention, N = 4096, d = 64 | 0.54 ms | `(Q * K') * V`: 22 ms (41×) | `Q * (K' * V)`: 0.51 ms |

The GP memory includes the kernel matrices (KernelFunctions.jl); the
Kalman memory is mostly the n × n result.  The expert Kalman filter
preallocates its buffers, works in place with `mul!` and solves with the
Cholesky factor of S (in the notebook); ulm.lam is faster because it also
exploits the symmetry of P: one triangle of F P Fᵀ + Q (`gemmtr`), and
P − K H P as a `syrk` downdate with W = (H P)ᵀ L⁻ᵀ.  From Python: GP regression with
n = 8000 is 1.4× faster than SciPy written by hand and 8× faster than NumPy
with `inv`; ridge regression 3–4× faster than with `inv`; a linear layer
trained by gradient descent, written as formulas, is as fast as PyTorch on
the same BLAS and 5× faster than PyTorch's autograd for a small layer.  On
a cluster node (96 threads, MKL), GP regression with n = 100 000 runs as
fast as the same steps written by hand with MKL.

## How it works

```
Julia / Python ─trace─▶ ulm_lam dialect ─normalise, infer properties─▶ kernel selection
  ─▶ ulm_lamk dialect (BLAS/LAPACK kernels on tensors) ─one-shot bufferization, workspace, info─▶
  memrefs ─EmitC─▶ C++ calling the BLAS ─▶ shared library ─▶ called without copying the arrays
```

- `ulm_lam` (`include/ulm_lam/UlmLamOps.td`): operations of linear algebra
  (`mul`, `add`, `transpose`, `inverse`, `assume`, …) on tensors whose
  types carry properties.  Normalisation and property inference are MLIR
  passes.
- Kernel selection derives the evaluation, in the spirit of Linnea
  (Barthels, Psarras and Bientinesi): it matches factorisations and kernels
  against the expression, costs them in flops (polynomials in the run-time
  sizes) and, where the best plan depends on sizes known only at run time,
  emits both plans behind an `scf.if`.  Operators that do not change inside
  a loop are hoisted out of it when the trip count makes that pay.
- `ulm_lamk` (`include/ulm_lam/UlmLamKernelOps.td`): `gemm`, `syrk`,
  `trsm`, `potrf`, `getrf`, `getrs`, … in destination-passing style, so
  that MLIR's one-shot bufferization updates results in place.  Further
  passes put the temporaries into one caller-provided workspace and turn
  failed factorisations into LAPACK-style `info` codes.
- A lowering to EmitC produces C++ that calls the BLAS.  A Julia matrix
  (column-major) is passed as its row-major transpose; the transposes
  become strides and BLAS transposition flags, never copies.

The design document [`doc/ulm-lam.pdf`](doc/ulm-lam.pdf) describes the
dialects, the derivations and the measurements, and compares ulm.lam with
Linnea (`examples/linnea/`).

## Installation

Developed on macOS (Apple silicon) with Homebrew's LLVM.  The generated
C++ is portable; it has also been built and run on Linux (GCC, MKL).

```sh
brew install llvm          # LLVM with MLIR 23
git clone https://github.com/michael-lehn/ulm-lam
cd ulm-lam
make plugin                # the MLIR dialects and passes, an mlir-opt plugin
```

Julia (tested with 1.13):

```sh
julia --project=notebook -e 'using Pkg; Pkg.instantiate()'
julia --project=notebook -e 'using Pluto; Pluto.run(notebook="notebook/ulm-lam.jl")'
```

or `] dev julia/UlmLam` in your own environment.  Python 3 (NumPy
optional): `PYTHONPATH=python python3 python/examples/gp.py`.

Environment: `LLVM_PREFIX` (default `brew --prefix llvm`), `ULM_LAM_BACKEND`
(`julia`, `ulmblas` or `cblas`), `ULMBLAS_DIR` (the ulmBLAS sources, for
`ulmblas`), `ULM_LAM_BLAS_LIBS` (link flags for `cblas`), `CXX`, `OPENMP=0`
(without OpenMP, Open Multi-Processing).

## Development

`make` builds and checks everything: the ODS (Operation Definition
Specification) files, the plugin, `ulm-lam-opt` and the pass tests
(`make check`: lit and FileCheck, `test/`), the hand-written design stages, the
examples in `examples/` (each against an independent evaluation; their
generated C++ is kept in `examples/*/generated/`), the Python tests, the
Julia tests and examples, and the document (LuaLaTeX).  `make gp
showcases torch` runs the Python showcases.

| Path | Content |
|---|---|
| `include/ulm_lam/*.td` | ODS of the `ulm_lam` and `ulm_lamk` dialects, interfaces, passes, the kernel catalogue |
| `lib/` | the mlir-opt plugin: both dialects, normalisation, property inference, kernel selection, workspace, info, lowering to EmitC |
| `tools/ulm-lam-opt.cpp` | `ulm-lam-opt`: mlir-opt with the ulm.lam dialects and passes (`build/bin/`) |
| `test/` | lit/FileCheck tests of the passes: normalisation, property inference, kernel selection, lowering |
| `scripts/ulm-lam-compile.sh` | the pipeline from `ulm_lam` to C++ (keeps every stage) |
| `include/ulm_lam/cblas_backend.h` | the CBLAS/LAPACK backend of the generated code |
| `julia/UlmLam/` | the Julia frontend (`@lam`, `@code_lam`) and runtime |
| `python/ulm/lam/` | the Python frontend and runtime |
| `notebook/` | the Pluto notebook and its environment |
| `examples/` | design stages, loops, workspaces, run-time sizes, inverses, Linnea's versions of the problems, the cluster runs |
| `python/examples/`, `julia/examples/` | the showcases |
| `doc/` | the design document |
| `doc/field-guide-companion/` | ulm.lam read against the theory of *Modern Compiler Concepts: A Field Guide*: which facts its types carry, where each one is assumed, verified, inferred or checked at run time, and what MLIR provides versus what ulm.lam implements (LuaLaTeX; examples and their outputs included) |

## What is open

In detail in the [design document](doc/ulm-lam.pdf), section "What is
open":

- **Kernel selection.** A measured cost model instead of flop counts
  (the idea of ATLAS: measure on the machine what cannot be predicted);
  fallbacks when a method fails, and a check mode that estimates the error
  of a result (a factorisation with `info = 0` can still be useless for
  what follows); a search over whole programs, as Linnea does; e-graphs;
  kernels described as data.
- **Evaluation.** The LAMP benchmark and Linnea's application problems.
- **Usability and reach.** `Pkg.add` without a local LLVM/MLIR and C++
  compiler; differentiation at the matrix level; a connection to
  Reactant.jl/Enzyme-JAX; a `torch.compile` backend; a GPU backend.

## Background

ulm.lam was started for two reasons.  The first is
[FLENS](https://github.com/michael-lehn/FLENS): it gave C++ a mathematical
notation for linear algebra, and its LAPACK reimplementation, FLENS-LAPACK
(verified bit for bit against LAPACK), was written twenty years ago by
mapping formulas onto BLAS calls by hand, with blood, sweat and tears.  The
work was so mechanical that I wanted to see a machine do it.  The second is
to learn MLIR in a domain I know well but which is outside the usual
mapping onto an instruction set: here the BLAS is the instruction set,
just not a typical one.

Much of ulm.lam's view of the problem comes from
[Linnea](https://github.com/HPAC/linnea) (Henrik Barthels, Christos
Psarras, Paolo Bientinesi), which solves the Linear Algebra Mapping
Problem by a graph search.

## Open questions

Questions that are less about implementation than about what the right
thing to do is (also discussed on the
[LLVM Discourse](https://discourse.llvm.org/t/mapping-numerical-linear-algebra-code-onto-blas-lapack/92808)).

**When are parentheses sacred?**  A compiler is greedy: `X = A * B * C`
gets parenthesised so that the flops are minimal, and ulm.lam does exactly
that (it flattens `A*(B*C)` into one chain and reorders it).  But sometimes
the parentheses were set for numerical reasons.  For `X = A*(B - C)`
versus `X = A*B - A*C` it is easy to say which needs fewer flops, but not
which is numerically stable: when `B` and `C` are close, the difference
`B - C` is computed exactly or nearly so, while `A*B - A*C` subtracts two
large, rounded products.  ulm.lam does not apply distributivity, so both
stay as written; in the [e-graph experiment](experiments/egraph) such a
rewrite would be a single rule, and then the question becomes urgent.  One
way out: "volatile" blocks, or attributes, that suppress certain rewrites
where the author fixed the order on purpose.

**What is the objective function?**  Flops are not the whole story.
Reimplementing LAPACK (FLENS-LAPACK, compared bit for bit against LAPACK)
showed how much more complex LAPACK's implementations of well-known
algorithms are than the textbook versions.  For eigenvalues, for example,
LAPACK has several variants, one slower but very robust, others fast but
less robust; after an iteration it computes a non-trivial error estimate
and chooses the variant for the next one.  A tiny difference in a result
can change the path the algorithm takes, even whether it converges.  A
factorisation can also succeed (`info = 0`) and still be useless for the
next step, e.g. the QR factorisation of a numerically rank-deficient
matrix.  What a compiler should optimise for, and what it must keep, is
open.

**Workspace: LAPACK style or MATLAB style?**  LAPACK's way is a workspace
given by the caller, with a query for its size (`lwork = -1`); ulm.lam
does the same, and the query accumulates over calls: if `foo` calls `bar`,
`foo`'s workspace contains `bar`'s part, and nothing is allocated inside
nested calls.  How much workspace there is depends on how the functions
are written.  The guideline: write the functions that do the work LAPACK
style, with `inout` parameters where a matrix may be overwritten and
results written into parameters (`update!` in Julia, `lam.update` in
Python), so that the workspace is minimal, often none at all; and put a
user-friendly MATLAB-style wrapper with read-only parameters on top.  The
copies, and the workspace for them, are then in the wrapper only:

```julia
@lam bar!(M::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) = update!(y, M \ y)
@lam foo!(M::InOut(Mat(:n, :n, :spd)), N::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) =
    update!(y, bar!(M, bar!(N, y)))
@lam foo(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = foo!(M, N, y)
```

`bar!` and `foo!` need no workspace; `foo` copies `y` directly into its
result and `M`, `N` into its workspace, and calls `foo!`, which is the
wrapper one would write by hand.  Written MATLAB style throughout, every
level pays for its own copies ([examples/calls](examples/calls), section
"Functions calling functions" of the design document).  Open is whether
the compiler should do more here: derive the LAPACK-style variant from a
MATLAB-style function itself, or warn where a copy could be avoided by an
`inout` parameter.
