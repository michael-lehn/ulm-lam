"""Write notebook/ulm-lam.jl (a Pluto notebook) from the cells below:

    python3 scripts/make_notebook.py notebook/ulm-lam.jl
"""
import uuid, sys
cells = []
def md(s): cells.append('md"""\n' + s.strip("\n") + '\n"""')
def code(s): cells.append(s.strip("\n"))

md(r'''
# ulm.lam: write the formula, get the BLAS/LAPACK calls an expert would write

ulm.lam (Ulm Linear Algebra Mapping) is a compiler built on MLIR (Multi-Level
Intermediate Representation).  You write linear algebra the way it is written
in a paper (`inv`, `'`, `*`, `\`, `+`) and declare what you know about the
matrices: symmetric positive definite, full rank, triangular, …

ulm.lam derives the sequence of BLAS (Basic Linear Algebra Subprograms) and
LAPACK (Linear Algebra PACKage) calls an expert would write by hand:

* factorisations and triangular solves instead of inverses,
* each factorisation computed once and reused,
* symmetry exploited (`syrk` instead of `gemm`, half the flops),
* the cheaper order of a product, decided at run time from the sizes,
* temporaries in one reused workspace, results written in place, loops without
  allocations.

`@code_lam` shows what it derived.  The generated code calls the BLAS Julia
uses (OpenBLAS, or anything libblastrampoline forwards to).

*An early research prototype; the sources are at
[github.com/michael-lehn/ulm-lam](https://github.com/michael-lehn/ulm-lam).*

**[▶ Run this notebook live](https://mybinder.org/v2/gh/michael-lehn/ulm-lam/master?urlpath=pluto%2Fopen%3Fpath%3D%2Fhome%2Fjovyan%2Fnotebook%2Fulm-lam.jl)** (on mybinder.org, with LLVM/MLIR,
Julia and ulm.lam in the container; starting takes a minute or two).  Change
a formula and `@code_lam` shows the new derivation.
''')
code(r'''
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end
''')
md(r'''
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
''')
code(r'''
begin
    measure = extern(:measure, [Index], Vec(:m))
    estimate = extern(:estimate, [Index, Vec(:n)])
    @lam function kalman(F::Mat(:n, :n), Q::Mat(:n, :n, :spsd), H::Mat(:m, :n),
                         R::Mat(:m, :m, :spd), x0::Vec(:n), P0::Mat(:n, :n, :spd),
                         steps::Index)
        loop(steps, x0, P0) do i, x, P
            P = assume(P, :symmetric)        # a covariance stays symmetric
            x = F * x                                       # predict
            P = F * P * F' + Q
            S = assume(H * P * H' + R, :spd)                # update
            K = P * H' * inv(S)
            x = x + K * (measure(i) - H * x)
            P = P - K * H * P
            estimate(i, x)
            (x, P)
        end
    end
end
''')
code(r'''
@code_lam kalman(F, Q, H, R, x0, P0, steps)
''')
md(r'''
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
''')
code(r'''
function model(n, m, steps)
    c, ν = 0.4, 0.2
    F = zeros(n, n)
    for i in 1:n
        F[i, i] = 1 - 2ν
        F[i, mod1(i + 1, n)] = ν - c / 2
        F[i, mod1(i - 1, n)] = ν + c / 2
    end
    d = [min(abs(i - j), n - abs(i - j)) for i in 1:n, j in 1:n]
    Q = 0.01 .* exp.(-d ./ 10)
    H = zeros(m, n)
    for (k, s) in enumerate(sort(randperm(n)[1:m]))
        H[k, s] = 1
    end
    R = 0.05 * Matrix(I, m, m)
    Lq = cholesky(Symmetric(Q + 1e-12I)).L
    truth = [sin(2π * i / n) + 0.5 * sin(6π * i / n) for i in 0:n-1]
    zs = map(1:steps) do _
        truth = F * truth + Lq * randn(n)
        H * truth + sqrt(0.05) * randn(m)
    end
    F, Q, H, R, zeros(n), Matrix(1.0I, n, n), zs, truth
end
''')
code(r'''
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;
''')
code(r'''
function kalman_literal(F, Q, H, R, x, P, zs)
    for z in zs
        x = F * x
        P = F * P * F' + Q
        S = H * P * H' + R
        K = P * H' * inv(S)
        x = x + K * (z - H * x)
        P = P - K * H * P
    end
    x, P
end
''')
code(r'''
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))
''')
code(r'''
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end
''')
md(r'''
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
''')
code(r'''
begin
    batch_x = extern(:batch_x, [Index], Mat(:B, :p))
    batch_y = extern(:batch_y, [Index], Mat(:B, :k))
    @lam function train(W0::Mat(:p, :k), b0::Vec(:k), ones::Vec(:B), step::Float,
                        steps::Index)
        loop(steps, W0, b0) do i, W, b
            X, Y = batch_x(i), batch_y(i)
            R = X * W + ones * b' - Y                       # forward pass, residual
            (W - step * (X' * R), b - step * (R' * ones))   # gradient step
        end
    end
end
''')
code(r'''
@code_lam train(W0, b0, ones, step, steps)
''')
md(r'''
Four BLAS calls per step and no allocation: the residual is computed into a
copy of ``Y`` (`gemm` with ``\beta = -1``, then the bias as a rank-1 `gemm`),
the update ``W - \text{step}\, X^T R`` is one `gemm` with
``\alpha = -\text{step}``, ``\beta = 1`` directly into ``W``, and the bias
update one `gemv`.  The batches Julia delivers are read in place
(column-major), without copies.  From Python, the same function is as fast as
PyTorch on the same BLAS and 5× faster than PyTorch's autograd for a small
layer (`python/examples/torch_linear.py`).
''')
code(r'''
begin
    Random.seed!(6)
    B, p, k, nsteps = 256, 1024, 256, 20
    Wtrue = randn(p, k) / sqrt(p)
    Xs = [randn(B, p) for _ in 1:nsteps]
    Ys = [X * Wtrue + 0.1 * randn(B, k) for X in Xs]
    # the same batches in Float32 (for the comparison below)
    Xs32, Ys32 = Matrix{Float32}.(Xs), Matrix{Float32}.(Ys)
    use32 = Ref(false)
    bind!(train; batch_x = i -> use32[] ? Xs32[i+1] : Xs[i+1],
                 batch_y = i -> use32[] ? Ys32[i+1] : Ys[i+1])
    W0, b0, onesB, lr_step = zeros(p, k), zeros(k), ones(B), 2 * 0.01 / B
end;
''')
code(r'''
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end
''')
code(r'''
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step))
''')
code(r'''
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end
''')
md(r'''
Machine learning mostly computes in single precision.  Called with `Float32`
arrays, a `@lam` function is compiled for `Float32` (a second library, as
Julia specialises on argument types): the same derivation on `sgemm`,
`spotrf`, …
''')
code(r'''
let (W32, b32, o32) = (Float32.(W0), Float32.(b0), Float32.(onesB))
    compare("ulm.lam, Float64" => () -> (use32[] = false;
                                         train(W0, b0, onesB, lr_step, nsteps)),
            "ulm.lam, Float32" => () -> (use32[] = true;
                                         train(W32, b32, o32, Float32(lr_step), nsteps)))
end
''')
md(r'''
## Inverses without inverses: generalised least squares

Generalised least squares: the best linear unbiased estimate of ``\beta`` in
``y = X\beta + \varepsilon`` with ``\operatorname{cov}(\varepsilon) = M``:

```math
\hat\beta = (X^T M^{-1} X)^{-1} X^T M^{-1} y
```

In Julia, with the properties as part of the types (`:n` and `:m` are sizes
known only at run time):
''')
code(r'''
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
''')
md(r'''
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
''')
code(r'''
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;
''')
code(r'''
β = gls(X, M, y)
''')
md(r'''
## Why it is fast: `@code_lam`
''')
code(r'''
@code_lam gls(X, M, y)
''')
md(r'''
What happened to the formula:

* `inv(M)` appears twice; ulm.lam factorises ``M = L_1 L_1^T`` once
  (`potrf`) and never forms an inverse.
* ``X^T M^{-1} X = (L_1^{-1}X)^T (L_1^{-1}X)``: one triangular solve (`trsm`)
  and a symmetric rank-k update (`syrk`), which computes one triangle only.
* The outer inverse becomes a second Cholesky factorisation and two triangular
  solves with a vector (`trsv`).
* `M` and `X` are not overwritten (the factor goes to a workspace that is
  reused across calls); the results of a factorisation that fails come back
  as `InfoError`, with the meaning listed at the end.

The same function written as plain Julia, and the version a numerical analyst
writes by hand:
''')
code(r'''
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y
''')
code(r'''
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end
''')
code(r'''
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))
''')
code(r'''
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result
''')
md(r'''
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
''')
code(r'''
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end
''')
code(r'''
@code_lam gp(K, s2, Ks, Kss, y)
''')
md(r'''
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
''')
code(r'''
begin
    kernel = SqExponentialKernel()
    s2 = 0.01
    xtrain = 10 .* rand(4000); xtest = 10 .* rand(1000)
    ytrain = sin.(xtrain) .+ 0.1 .* randn(4000)
    gp_ulmlam(x, xs, y) =
        gp(kernelmatrix(kernel, x), s2, kernelmatrix(kernel, x, xs),
           kernelmatrix(kernel, xs), y)
    gp_abstractgps(x, xs, y) = mean_and_cov(posterior(GP(kernel)(x, s2), y)(xs))
end;
''')
code(r'''
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))
''')
code(r'''
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end
''')
md(r'''
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
''')
md(r'''
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
''')
code(r'''
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V
''')
code(r'''
@code_lam attention(Q, K, V)
''')
md(r'''
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
''')
code(r'''
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end
''')
md(r'''
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
''')
code(r'''
BLAS.get_config()
''')
md(r'''
With [UlmBLAS.jl](https://github.com/michael-lehn/UlmBLAS.jl),
`using UlmBLAS; UlmBLAS.forward()` makes ulmBLAS the BLAS of Julia, and with
it of every ulm.lam function.  The generated code can also call the
header-only C++ library ulmBLAS directly (`UlmLam.compile(f; backend=:ulmblas)`)
or any CBLAS (C interface to the BLAS) and LAPACK, e.g. MKL (Intel Math
Kernel Library) on a cluster.  The same functions are available from Python
with NumPy arrays.

## Under the hood: MLIR

`@lam` traces the Julia function into ulm.lam's own MLIR dialect: operations
of linear algebra on matrices whose types carry the properties.
''')
code(r'''
Text(first(UlmLam.lam_module([gls])))
''')
md(r'''
MLIR passes then

1. normalise the expression and infer properties (``X^T M^{-1} X`` is
   symmetric positive definite because ``M`` is and ``X`` has full rank),
2. select kernels with a cost model (flops, and which sizes are only known at
   run time; then both plans are generated with a run-time `if`),
   producing a second dialect of BLAS/LAPACK-level kernels on tensors,
3. bufferise (one-shot bufferization), hoist allocations out of loops, place
   temporaries in one workspace and turn the results of factorisations into
   LAPACK-style `info` codes,
4. lower to EmitC: C++ calling the BLAS and LAPACK, compiled to a shared
   library that Julia calls with `ccall`, without copying the arrays.

## What you can write

| | |
|:--|:--|
| matrices | `Mat(rows, cols, props...)`; sizes are numbers or symbols (run-time sizes) |
| vectors | `Vec(n)` |
| scalars | `Float` (a run-time scalar), numbers in the formula |
| integers | `Index` (trip counts, arguments of callbacks) |
| properties | `:symmetric`, `:spd`, `:spsd`, `:lower_tri`, `:upper_tri`, `:unit_diag`, `:diagonal`, `:full_rank`, `:non_singular`, `:orthogonal`, `:permutation`, `:identity`, `:zero` |
| operations | `*`, `+`, `-`, `'`, `transpose`, `inv`, `\`, `/`, scalar `*` and `/`, `I`, `eye(n)` |
| facts | `assume(A, props...)` for what the compiler cannot derive |
| control | `loop(n, carry...) do i, x... end`, several results as a tuple |
| Julia code | `extern(name, argtypes, result)` and `bind!(f; name = g)` |
| in place | `InOut(Mat(...))`: a parameter the function may overwrite |
| inspect | `@code_lam f(args...)` |

In Python the same with `@lam.function`, `Matrix("n", "m", "spd")`, `@`,
`.T`, `.inv`, `lam.eye`, `lam.assume`, `lam.loop`, `lam.extern`.

So far: real double precision, dense matrices.
''')

md(r'''
## Appendix: the timing helper
''')
code(r'''
"Time (best of several runs) and memory allocated per call, relative to the first case."
function compare(cases...)
    rows = map(cases) do (name, f)
        f()
        t = @belapsed $f() samples = 5 evals = 1 seconds = 20
        (name, t, @allocated f())
    end
    t0 = rows[1][2]
    s = "| | time | relative | allocated |\n|:--|--:|--:|--:|\n"
    for (name, t, a) in rows
        s *= @sprintf("| %s | %.2f ms | %.1f× | %s |\n", name, 1e3t, t / t0,
                      Base.format_bytes(a))
    end
    Markdown.parse(s)
end
''')

ids = [str(uuid.uuid4()) for _ in cells]
out = ["### A Pluto.jl notebook ###", "# v0.20.21", "", "using Markdown", "using InteractiveUtils", "",
       "# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).",
       "macro bind(def, element)",
       "    #! format: off",
       "    return quote",
       "        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID(\"6e696c72-6542-2067-7265-42206c756150\"), \"AbstractPlutoDingetjes\")].Bonds.initial_value catch; b -> missing; end",
       "        local el = $(esc(element))",
       "        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)",
       "        el",
       "    end",
       "    #! format: on",
       "end", ""]
for i, c in zip(ids, cells):
    out += [f"# ╔═╡ {i}", c, ""]
out += ["# ╔═╡ Cell order:"] + [f"# ╟─{i}" if c.startswith('md"""') else f"# ╠═{i}" for i, c in zip(ids, cells)]
open(sys.argv[1], "w").write("\n".join(out) + "\n")
