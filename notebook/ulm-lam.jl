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

# ╔═╡ 16268b8f-b8fa-4e46-a6de-9c5e56afd3fd
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

# ╔═╡ 4f6d1ae5-2af7-4412-a9f5-b1019df25d82
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 5ead4f2a-734c-4759-b9c3-e5fb4dc78632
md"""
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
"""

# ╔═╡ 884430cc-605e-45ae-b9d6-1c6878eeffd2
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

# ╔═╡ efe97fdd-25dc-4fc6-9664-29829cf3b7d4
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ be87aea7-1887-41af-9cd2-d6ea8e34b420
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
"""

# ╔═╡ 39b577c3-c16b-4b35-8d97-14ffc9a406d6
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

# ╔═╡ b302aaaa-dfcc-48b5-809a-5c05a0ec624b
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ bd3e7c60-a44e-49f3-82f7-1cc834bba807
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

# ╔═╡ bdf4ac0c-22a9-4c18-8578-3e4021bafcea
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))

# ╔═╡ 5bdb404d-8fa7-44ef-bb28-61bfc52e0120
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end

# ╔═╡ 88ac0867-3ae1-4ce9-812d-61b3651ce352
md"""
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
"""

# ╔═╡ 9cbe2f25-8d05-4903-825f-30ae94468614
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

# ╔═╡ 23b6985d-9c73-482a-b187-7507f4203739
@code_lam train(W0, b0, ones, step, steps)

# ╔═╡ eb466485-bcbc-4d13-843c-06ca3747c98d
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

# ╔═╡ 4b2fc376-ba94-4d34-9993-1af6931f395c
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

# ╔═╡ a7460ce3-aa7d-44af-95c7-d5fdc768d42a
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end

# ╔═╡ 1a6b9113-cad5-47ed-99f6-7cb12a7a377b
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step))

# ╔═╡ a51f4ee9-d5f8-4f9a-8cf9-48627f1822d0
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end

# ╔═╡ e09fcb72-f8a6-428b-aa8d-490c882b9b26
md"""
Machine learning mostly computes in single precision.  Called with `Float32`
arrays, a `@lam` function is compiled for `Float32` (a second library, as
Julia specialises on argument types): the same derivation on `sgemm`,
`spotrf`, …
"""

# ╔═╡ 2dac1850-21e1-4c7f-a246-40c613fe33e9
let (W32, b32, o32) = (Float32.(W0), Float32.(b0), Float32.(onesB))
    compare("ulm.lam, Float64" => () -> (use32[] = false;
                                         train(W0, b0, onesB, lr_step, nsteps)),
            "ulm.lam, Float32" => () -> (use32[] = true;
                                         train(W32, b32, o32, Float32(lr_step), nsteps)))
end

# ╔═╡ 4cd0a82f-bb6c-4be8-a712-61fb8ed3a89b
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

# ╔═╡ 6b686303-844e-421b-98b9-a06d275098a2
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ 0739acbc-de7a-42b8-9a63-798158076103
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ e28aa756-c396-4b23-ac4c-4ff20a6472bc
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ 132de8ef-ad45-4336-a383-70170f2c0af0
β = gls(X, M, y)

# ╔═╡ 1680d5ed-a7ba-41bd-bd6c-5406d1b9b58a
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ 23a68b58-50e2-415f-a37b-bfb4b3a4a66a
@code_lam gls(X, M, y)

# ╔═╡ 4a96a54d-ff42-4a42-b06b-624a26d90417
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

# ╔═╡ 39f93c8a-d05e-478f-b553-bd9168e25fb1
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ ee593e69-2d9b-43ba-9d18-ca277aa732e1
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ cf0bf34e-6324-458f-9b4b-bf279c18719e
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ f1dbeef9-c55f-4866-8e5c-4c05ceba4a67
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ c7abdff7-de05-47f5-b22b-a429b7c14b70
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ d7192e4c-679f-42b3-ac1b-2b757fec7030
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 0e1ae73a-3a2c-4d4c-9357-4e05ea2ef237
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ fedd05fd-771d-4c45-828f-bd5ea7abf9a4
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ 1b95ee40-b587-4922-97a9-7112f55df7fe
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

# ╔═╡ e4f8f9f5-460a-45ad-9a18-7dd4d3b7a386
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ 2609c88e-737d-4f26-aa95-b403ac701c97
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ c9edeb29-eaa1-467e-ac80-23f365c12ae8
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
"""

# ╔═╡ 5687547c-94f6-4d4e-b7dc-a6e674913d4d
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ 4f8bfc81-1e7c-4935-a477-bd2d33f27712
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ e3565540-7d97-4c0f-831a-bf144993cc6d
@code_lam attention(Q, K, V)

# ╔═╡ be42dcdd-99aa-4685-bc2b-57ed56ea4af3
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ 826077a7-97ba-443e-8650-2d91f944447f
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ bd2ba14d-c4e4-4293-ae99-406671716164
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ 39303d46-9168-4707-90c9-bf5e60c2deeb
BLAS.get_config()

# ╔═╡ 4928d7b2-9b12-401f-94fb-d682fb606bd0
md"""
With [UlmBLAS.jl](https://github.com/michael-lehn/UlmBLAS.jl) (`Pkg.add("UlmBLAS")`),
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

# ╔═╡ 51746610-2d42-4014-9ac1-0e7a3f4f28a0
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ 20718fbb-ef67-482c-be54-d3d69a8206bb
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

# ╔═╡ 861db077-cb91-49e6-ab62-11f7d898098b
md"""
## Appendix: the timing helper
"""

# ╔═╡ cccc08d9-da98-47d9-a09a-72e9b960839b
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
# ╟─16268b8f-b8fa-4e46-a6de-9c5e56afd3fd
# ╠═4f6d1ae5-2af7-4412-a9f5-b1019df25d82
# ╟─5ead4f2a-734c-4759-b9c3-e5fb4dc78632
# ╠═884430cc-605e-45ae-b9d6-1c6878eeffd2
# ╠═efe97fdd-25dc-4fc6-9664-29829cf3b7d4
# ╟─be87aea7-1887-41af-9cd2-d6ea8e34b420
# ╠═39b577c3-c16b-4b35-8d97-14ffc9a406d6
# ╠═b302aaaa-dfcc-48b5-809a-5c05a0ec624b
# ╠═bd3e7c60-a44e-49f3-82f7-1cc834bba807
# ╠═bdf4ac0c-22a9-4c18-8578-3e4021bafcea
# ╠═5bdb404d-8fa7-44ef-bb28-61bfc52e0120
# ╟─88ac0867-3ae1-4ce9-812d-61b3651ce352
# ╠═9cbe2f25-8d05-4903-825f-30ae94468614
# ╠═23b6985d-9c73-482a-b187-7507f4203739
# ╟─eb466485-bcbc-4d13-843c-06ca3747c98d
# ╠═4b2fc376-ba94-4d34-9993-1af6931f395c
# ╠═a7460ce3-aa7d-44af-95c7-d5fdc768d42a
# ╠═1a6b9113-cad5-47ed-99f6-7cb12a7a377b
# ╠═a51f4ee9-d5f8-4f9a-8cf9-48627f1822d0
# ╟─e09fcb72-f8a6-428b-aa8d-490c882b9b26
# ╠═2dac1850-21e1-4c7f-a246-40c613fe33e9
# ╟─4cd0a82f-bb6c-4be8-a712-61fb8ed3a89b
# ╠═6b686303-844e-421b-98b9-a06d275098a2
# ╟─0739acbc-de7a-42b8-9a63-798158076103
# ╠═e28aa756-c396-4b23-ac4c-4ff20a6472bc
# ╠═132de8ef-ad45-4336-a383-70170f2c0af0
# ╟─1680d5ed-a7ba-41bd-bd6c-5406d1b9b58a
# ╠═23a68b58-50e2-415f-a37b-bfb4b3a4a66a
# ╟─4a96a54d-ff42-4a42-b06b-624a26d90417
# ╠═39f93c8a-d05e-478f-b553-bd9168e25fb1
# ╠═ee593e69-2d9b-43ba-9d18-ca277aa732e1
# ╠═cf0bf34e-6324-458f-9b4b-bf279c18719e
# ╠═f1dbeef9-c55f-4866-8e5c-4c05ceba4a67
# ╟─c7abdff7-de05-47f5-b22b-a429b7c14b70
# ╠═d7192e4c-679f-42b3-ac1b-2b757fec7030
# ╠═0e1ae73a-3a2c-4d4c-9357-4e05ea2ef237
# ╟─fedd05fd-771d-4c45-828f-bd5ea7abf9a4
# ╠═1b95ee40-b587-4922-97a9-7112f55df7fe
# ╠═e4f8f9f5-460a-45ad-9a18-7dd4d3b7a386
# ╠═2609c88e-737d-4f26-aa95-b403ac701c97
# ╟─c9edeb29-eaa1-467e-ac80-23f365c12ae8
# ╟─5687547c-94f6-4d4e-b7dc-a6e674913d4d
# ╠═4f8bfc81-1e7c-4935-a477-bd2d33f27712
# ╠═e3565540-7d97-4c0f-831a-bf144993cc6d
# ╟─be42dcdd-99aa-4685-bc2b-57ed56ea4af3
# ╠═826077a7-97ba-443e-8650-2d91f944447f
# ╟─bd2ba14d-c4e4-4293-ae99-406671716164
# ╠═39303d46-9168-4707-90c9-bf5e60c2deeb
# ╟─4928d7b2-9b12-401f-94fb-d682fb606bd0
# ╠═51746610-2d42-4014-9ac1-0e7a3f4f28a0
# ╟─20718fbb-ef67-482c-be54-d3d69a8206bb
# ╟─861db077-cb91-49e6-ab62-11f7d898098b
# ╠═cccc08d9-da98-47d9-a09a-72e9b960839b
