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
def control(A: Matrix("n", "n"), steps: Index):
    with lam.loop(steps) as L:
        b = get_sensor_data(L.index)
        x = A.inv @ b
        react(L.index, x)

lam.emit_mojo(control, "sensor_lam.mojo")
```

The generated [`sensor_lam.mojo`](examples/sensor/sensor_lam.mojo)
factorizes A once, before the loop (`K.getrf`), and in the loop solves on
b in place (`K.getrs`: the row swaps and the two triangular solves):

```mojo
var v44: Int = K.getrf(v1, v1, v40, v1, 1, v38, 1)
...
for v50 in range(v14, v3, v7):
    ext.get_sensor_data(v1, v50, v49)
    _ = K.getrs(
        False, False, v1, 1, v40, v1, 1, v38, 1, v49, 1, 1
    )
    ext.react(v1, v50, v49)
```

The application implements the external functions as a Mojo struct that
conforms to the generated trait `Externs`
([`examples/sensor/main.mojo`](examples/sensor/main.mojo)):

```mojo
from sensor_lam import Externs, control

struct Sensor(Externs):
    def get_sensor_data(mut self, n: Int, l: Int, b: P[DType.float64]): ...
    def react(mut self, n: Int, l: Int, x: P[DType.float64]): ...
```

```
$ pixi run sensor
n = 1000  steps = 200  (Apple Accelerate)
  ulm.lam (getrf once, getrs per step)  85 ms  info 0  max error 7.0e-15
  as written (inv(A) * b per step)      2078 ms           max error 6.9e-15
  speedup 24.5
```

(Apple M5; "as written" computes inv(A) and the product in every step,
with the same Accelerate calls.)

## Using the generated code

- **Build**: `mojo build -I <ulm-lam>/mojo -Xlinker -framework -Xlinker
  Accelerate main.mojo` (the package `ulm_lam` and Accelerate).
- **Calling convention**: the comment in front of every generated function
  lists its parameters.  Matrices are raw pointers
  (`P[DType.float64]`, `ulm_lam.kernels.P`) to row-major arrays the caller
  owns; sizes are `Int`.  Mojo frees a `List` after its last use, so keep
  the lists alive while the function runs (`_ = A^` after the call).
- **Workspace**: the last two parameters are the workspace (bytes) and
  `info`.  Call once with `info[0] = -1` (a query: the size in bytes is
  returned in `info[0]`), allocate it once, then call with `info[0]` set
  to that size, as often as needed: no allocation in the call.  (`info[0]
  = 0` lets the function allocate it itself.)  The result is LAPACK's
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
