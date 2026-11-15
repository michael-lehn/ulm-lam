# ulm.lam for Mojo

ulm.lam (Ulm Linear Algebra Mapping) generates
[Mojo](https://www.modular.com/mojo): you write the linear algebra as on
paper; `ulm-lam --emit=mojo` writes the Mojo an expert would write: BLAS
(Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage)
calls, factorizations instead of inverses, computed once, outside loops,
temporaries in one workspace.  The calls go to Apple Accelerate for now,
through the kernels in [`ulm_lam/kernels.mojo`](ulm_lam/kernels.mojo).

## The example of my lectures

An application reads sensor data b in every step and has to solve
A x = b.  Close to the algorithm ([`examples/sensor/gen.py`](examples/sensor/gen.py),
ulm.lam's Python frontend):

```python
get_sensor_data = lam.extern("get_sensor_data", [Index], Vector("n"))
react = lam.extern("react", [Index, Vector("n")])

@lam.function
def control(A: InOut(Matrix("n", "n")), steps: Index):
    with lam.loop(steps) as L:
        b = get_sensor_data(L.index)
        x = A.inv @ b
        react(L.index, x)

@lam.function
def control_keep(A: Matrix("n", "n"), steps: Index):
    control(A, steps)

lam.emit_mojo([control, control_keep], "sensor_lam.mojo")
```

The generated [`sensor_lam.mojo`](examples/sensor/sensor_lam.mojo)
factorizes A once, before the loop (`K.getrf`), and in the loop solves on
b in place (`K.getrs`: the row swaps and the two triangular solves):

```mojo
var v35: Int = K.getrf(v1, v1, v2, v1, 1, v34, 1)
...
for v41 in range(v14, v3, v7):
    ext.get_sensor_data(v1, v41, v40)
    _ = K.getrs(
        False, False, v1, 1, v2, v1, 1, v34, 1, v40, 1, 1
    )
    ext.react(v1, v41, v40)
```

As in LAPACK, `control` overwrites A with its LU (lower-upper) factors
(`InOut`): no copy, and its workspace holds only the pivots and b (12 KB
for n = 1000).  `control_keep` is the wrapper for a caller that still
needs A: it copies A into its own workspace (8 MB more) and calls
`control`.  A matrix parameter without `InOut` is read only, and the
function factorizes a copy.

The application implements the external functions as a Mojo struct that
conforms to the generated trait `Externs`
([`examples/sensor/main.mojo`](examples/sensor/main.mojo)):

```mojo
from sensor_lam import Externs, control, control_keep

struct Sensor(Externs):
    def get_sensor_data(mut self, n: Int, l: Int, b: P[DType.float64]): ...
    def react(mut self, n: Int, l: Int, x: P[DType.float64]): ...
```

```
$ pixi run sensor
n = 1000  steps = 200  (Apple Accelerate)
  ulm.lam control      (A overwritten)  59 ms  info 0  max error 7.0e-15  workspace 12032 bytes
  ulm.lam control_keep (A kept)         59 ms  info 0  max error 7.0e-15  workspace 8012032 bytes
  as written (inv(A) * b per step)    1455 ms          max error 6.9e-15
  speedup 24.7
```

(Apple M5; "as written" computes inv(A) and the product in every step,
with the same Accelerate calls.)

## Using the generated code

- **Build**: `mojo build -I <ulm-lam>/mojo -Xlinker -framework -Xlinker
  Accelerate main.mojo` (the package `ulm_lam` and Accelerate).
- **Calling convention**: the comment in front of every generated function
  lists its parameters.  Matrices are raw pointers
  (`P[DType.float64]`, `ulm_lam.kernels.P`) to row-major arrays the caller
  owns; sizes are `Int`.  A parameter marked `inout` in the comment may
  be overwritten (LAPACK's "on exit, A is destroyed").  Mojo frees a `List` after its last use, so keep
  the lists alive while the function runs (`_ = A^` after the call).
- **Workspace**: the last two parameters are the workspace (bytes) and
  `info`.  Call once with `info[0] = -1` (a query: the size in bytes is
  returned in `info[0]`), allocate it once, then call with `info[0]` set
  to that size, as often as needed: no allocation in the call.  (`info[0]
  = 0` lets the function allocate it itself, in every call: a
  convenience, not for loops.)  The result is LAPACK's
  `info`: 0 on success, otherwise what the comment lists (a singular
  matrix, a matrix that is not positive definite, ...).
- **External functions**: the methods of the trait `Externs`; every
  function of the program takes the caller's implementation first
  (`control(sensor, n, A, steps, work, info)`).
- **Element type**: `lam.emit_mojo(..., dtype="float32")` for `Float32`.

## What is here

| | |
|---|---|
| `ulm_lam/kernels.mojo` | the kernels of the generated code on Accelerate: the Mojo port of `julia/UlmLAM/src/kernels.jl` (strided views as transposes, flipped triangles or the transposed problem, no copies); CBLAS (the C interface to the BLAS) with 32-bit integers, LAPACK under its `$NEWLAPACK` symbols |
| `tools/ulm-lam.cpp` (`MojoEmitter`) | `ulm-lam --emit=mojo`: the program after the lowering to EmitC as a Mojo module; `--mojo-c-abi` also exports every function with the C calling convention |
| `python/ulm/lam/runtime.py` | `lam.emit_mojo`, and `lam.compile(..., backend="mojo")` (or `ULM_LAM_BACKEND=mojo`): the Mojo as a shared library, called from Python |
| `tests/test_kernels.mojo` | every kernel against plain loops, float64 and float32, row- and column-major |
| `tests/gls_driver.mojo` | GLS (generalized least squares) from `examples/gls`, generated, against NumPy |
| `check.sh` | `make mojo`: the kernels, GLS, the 55 Python tests with the Mojo backend, the sensor example |

## What is open

- Only Apple Accelerate, only macOS on Apple silicon (the pixi
  environment).  The next step: the native Mojo kernels of
  [mojo-gemm](https://github.com/michael-lehn/mojo-gemm) (its GEMM (general
  matrix-matrix multiply) and Cholesky factorization use the same stride
  convention), replacing Accelerate one call at a time.
- A few kernels allocate a temporary: `symm` with B and C in different
  orientations, `getrs`, `sytrs` and `gels` with a row-major right-hand
  side, `potri` of an upper factor, and the workspaces LAPACK's `getri`,
  `gels` and `sytrf` ask for.
- A frontend in Mojo itself: the program is written in Python today (or
  in Julia, or as MLIR (Multi-Level Intermediate Representation)).
