### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 062e7f5c-4f0c-4974-a498-8b6dc19a9bbe
md"""
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
"""

# ╔═╡ 45a6878a-2e08-48d3-bec9-3b9f18e6538a
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 36727dbf-aee6-4fe4-839d-0a7b27c13ce4
md"""
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
"""

# ╔═╡ cfa62102-7d6e-42b0-9f63-fa95af7f4709
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

# ╔═╡ 88e63e8a-56fb-4103-91eb-542dbe608024
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ b70300d4-4823-4964-a45c-1cb0538138b7
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
"""

# ╔═╡ f9ca6326-64b7-4bc2-a1db-f77b0f46a1a2
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

# ╔═╡ a445d40d-ee7e-40f0-b936-3e9835f5682a
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ 02770529-bcd3-4d9e-a211-28027a6d89fd
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

# ╔═╡ 3c67cf83-c636-4078-bed6-01fde7fb1cce
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))

# ╔═╡ c693df55-9bf8-4304-b41b-23f8395dcd07
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end

# ╔═╡ 72d472b9-1e52-4377-ac44-77c1d0952927
md"""
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
"""

# ╔═╡ 079eab95-17dd-4755-bea4-30c28240b833
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

# ╔═╡ 6b1e9511-b7e9-460e-bc92-bb97b1193d05
@code_lam train(W0, b0, ones, step, steps)

# ╔═╡ 5eeeab01-17b3-4c4d-a788-5f026fb28586
md"""
Four BLAS calls per step and no allocation: the residual is computed into a
copy of ``Y`` (`gemm` with ``\beta = -1``, then the bias as a rank-1 `gemm`),
the update ``W - \text{step}\, X^T R`` is one `gemm` with
``\alpha = -\text{step}``, ``\beta = 1`` directly into ``W``, and the bias
update one `gemv`.  The batches Julia delivers are read in place
(column-major), without copies.  From Python, the same function is as fast as
PyTorch on the same BLAS and 5× faster than PyTorch's autograd for a small
layer (`python/examples/torch_linear.py`).
"""

# ╔═╡ 8497e9c5-6711-4b81-b04a-884515c3b14b
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

# ╔═╡ 0ee3a20a-27af-4877-af1e-6a69d1ad4f09
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end

# ╔═╡ 0e6f9e53-b0a2-4e46-b44b-c33b5982caae
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step))

# ╔═╡ 3fcbe981-8a2e-4ccd-920e-d9206ef640dd
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end

# ╔═╡ 3a792fe9-4d2c-4f1a-8913-d0eeb00459e5
md"""
Machine learning mostly computes in single precision.  Called with `Float32`
arrays, a `@lam` function is compiled for `Float32` (a second library, as
Julia specialises on argument types): the same derivation on `sgemm`,
`spotrf`, …
"""

# ╔═╡ 08173a9d-3b7f-4007-8f1a-72ef9590ec50
let (W32, b32, o32) = (Float32.(W0), Float32.(b0), Float32.(onesB))
    compare("ulm.lam, Float64" => () -> (use32[] = false;
                                         train(W0, b0, onesB, lr_step, nsteps)),
            "ulm.lam, Float32" => () -> (use32[] = true;
                                         train(W32, b32, o32, Float32(lr_step), nsteps)))
end

# ╔═╡ 2ab5cdb2-d413-4749-8d0e-6f89d891c590
md"""
## Inverses without inverses: generalised least squares

Generalised least squares: the best linear unbiased estimate of ``\beta`` in
``y = X\beta + \varepsilon`` with ``\operatorname{cov}(\varepsilon) = M``:

```math
\hat\beta = (X^T M^{-1} X)^{-1} X^T M^{-1} y
```

In Julia, with the properties as part of the types (`:n` and `:m` are sizes
known only at run time):
"""

# ╔═╡ 139f93cf-e38b-4e89-9dbc-319a699d3419
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ 35f9be14-64e8-4bbd-9e17-8cadb420b8ec
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ f3f6c640-2fbb-46b2-9a92-535db0562b81
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ 35700ae5-b8b8-4079-ad8d-7dcd1f6f9475
β = gls(X, M, y)

# ╔═╡ fc4e47f1-3c15-419e-808b-824bedb4bf98
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ 2949c0d1-58fc-4f0c-b35b-6ef8fd4a8545
@code_lam gls(X, M, y)

# ╔═╡ f66abade-754c-4b52-96fe-83c9eca1b32a
md"""
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
"""

# ╔═╡ 83569672-03e2-45a7-b083-5d54dbc5602c
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ 76844b40-0a2f-43ae-b452-31f90a23d4de
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ ed69f76c-b99d-4541-aa64-ef7ba25135ad
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ 0f657bcf-4930-4bb1-ae36-ea63de0ecd00
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ 596617d1-300b-4efd-b879-c6ba9b6d04f2
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ 9b3ab64f-8dca-407b-b5d1-6990cee3ebe3
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 5254763b-b1de-4fde-8be9-9d3bcc61cc79
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ 1eca5a00-58fa-4a8f-b887-2792e7cbc798
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ 1918ffd9-6245-4f30-8ddf-b498ee6f1a23
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

# ╔═╡ e069d1fc-efe5-46cd-8918-f78e99e8660f
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ 52c10f8f-7bd1-44ef-87eb-af3341bcd1b9
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ cec1e190-cad8-458c-bcb4-06a062c6b66a
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
"""

# ╔═╡ cf8b1f1b-973e-439e-8886-f0a1da2d0fc4
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ cfe890cc-da0e-4f72-97a6-043722c783a6
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ 6e881e21-a430-49ee-8f87-99b41dce43e5
@code_lam attention(Q, K, V)

# ╔═╡ 75f58319-ecc8-46de-8df7-c1e1f3fb0ead
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ 4a77c282-81fe-4ac8-8e4d-469adb3dba61
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ 2df59cef-d808-4a60-8d2f-22f5bde2ad0d
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ 207892b7-fde8-4c05-bb47-154ba3c0c5ce
BLAS.get_config()

# ╔═╡ d8433f78-d56a-43d4-86a1-2f6673b54473
md"""
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
"""

# ╔═╡ e328dac1-6429-4fcc-aa4a-0beb207ac3c3
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ a1161405-f77a-47f2-85cb-4543e72d2587
md"""
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
"""

# ╔═╡ 660c3ee9-c0e4-4789-a208-f47886a6db4b
md"""
## Appendix: the timing helper
"""

# ╔═╡ 73df7f0c-a868-46ed-9b14-413495acfe63
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

# ╔═╡ Cell order:
# ╟─062e7f5c-4f0c-4974-a498-8b6dc19a9bbe
# ╠═45a6878a-2e08-48d3-bec9-3b9f18e6538a
# ╟─36727dbf-aee6-4fe4-839d-0a7b27c13ce4
# ╠═cfa62102-7d6e-42b0-9f63-fa95af7f4709
# ╠═88e63e8a-56fb-4103-91eb-542dbe608024
# ╟─b70300d4-4823-4964-a45c-1cb0538138b7
# ╠═f9ca6326-64b7-4bc2-a1db-f77b0f46a1a2
# ╠═a445d40d-ee7e-40f0-b936-3e9835f5682a
# ╠═02770529-bcd3-4d9e-a211-28027a6d89fd
# ╠═3c67cf83-c636-4078-bed6-01fde7fb1cce
# ╠═c693df55-9bf8-4304-b41b-23f8395dcd07
# ╟─72d472b9-1e52-4377-ac44-77c1d0952927
# ╠═079eab95-17dd-4755-bea4-30c28240b833
# ╠═6b1e9511-b7e9-460e-bc92-bb97b1193d05
# ╟─5eeeab01-17b3-4c4d-a788-5f026fb28586
# ╠═8497e9c5-6711-4b81-b04a-884515c3b14b
# ╠═0ee3a20a-27af-4877-af1e-6a69d1ad4f09
# ╠═0e6f9e53-b0a2-4e46-b44b-c33b5982caae
# ╠═3fcbe981-8a2e-4ccd-920e-d9206ef640dd
# ╟─3a792fe9-4d2c-4f1a-8913-d0eeb00459e5
# ╠═08173a9d-3b7f-4007-8f1a-72ef9590ec50
# ╟─2ab5cdb2-d413-4749-8d0e-6f89d891c590
# ╠═139f93cf-e38b-4e89-9dbc-319a699d3419
# ╟─35f9be14-64e8-4bbd-9e17-8cadb420b8ec
# ╠═f3f6c640-2fbb-46b2-9a92-535db0562b81
# ╠═35700ae5-b8b8-4079-ad8d-7dcd1f6f9475
# ╟─fc4e47f1-3c15-419e-808b-824bedb4bf98
# ╠═2949c0d1-58fc-4f0c-b35b-6ef8fd4a8545
# ╟─f66abade-754c-4b52-96fe-83c9eca1b32a
# ╠═83569672-03e2-45a7-b083-5d54dbc5602c
# ╠═76844b40-0a2f-43ae-b452-31f90a23d4de
# ╠═ed69f76c-b99d-4541-aa64-ef7ba25135ad
# ╠═0f657bcf-4930-4bb1-ae36-ea63de0ecd00
# ╟─596617d1-300b-4efd-b879-c6ba9b6d04f2
# ╠═9b3ab64f-8dca-407b-b5d1-6990cee3ebe3
# ╠═5254763b-b1de-4fde-8be9-9d3bcc61cc79
# ╟─1eca5a00-58fa-4a8f-b887-2792e7cbc798
# ╠═1918ffd9-6245-4f30-8ddf-b498ee6f1a23
# ╠═e069d1fc-efe5-46cd-8918-f78e99e8660f
# ╠═52c10f8f-7bd1-44ef-87eb-af3341bcd1b9
# ╟─cec1e190-cad8-458c-bcb4-06a062c6b66a
# ╟─cf8b1f1b-973e-439e-8886-f0a1da2d0fc4
# ╠═cfe890cc-da0e-4f72-97a6-043722c783a6
# ╠═6e881e21-a430-49ee-8f87-99b41dce43e5
# ╟─75f58319-ecc8-46de-8df7-c1e1f3fb0ead
# ╠═4a77c282-81fe-4ac8-8e4d-469adb3dba61
# ╟─2df59cef-d808-4a60-8d2f-22f5bde2ad0d
# ╠═207892b7-fde8-4c05-bb47-154ba3c0c5ce
# ╟─d8433f78-d56a-43d4-86a1-2f6673b54473
# ╠═e328dac1-6429-4fcc-aa4a-0beb207ac3c3
# ╟─a1161405-f77a-47f2-85cb-4543e72d2587
# ╟─660c3ee9-c0e4-4789-a208-f47886a6db4b
# ╠═73df7f0c-a868-46ed-9b14-413495acfe63
