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

# ╔═╡ a0949dc6-2754-4ce3-ae1e-f97f8fe4aa58
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

# ╔═╡ 1d316dc3-c5e5-4418-bd2b-44d4e84c9af8
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 0fd7bccd-e170-45b8-9989-53dc27c24eb9
md"""
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
"""

# ╔═╡ 9067af62-8c37-4d1d-aa0b-746dd87390d2
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

# ╔═╡ 6942d067-acbb-4b3b-9a59-6c7f7854be60
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ d472c25a-0785-48c6-96e1-2d2f06936cbe
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
"""

# ╔═╡ a86a0e3c-75ca-47c9-83eb-a36f151211d9
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

# ╔═╡ 090386b3-cffa-41e4-a237-6ead5ba90f18
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ f155547f-b2ba-46b7-94c5-664bb6f14303
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

# ╔═╡ 139ebbaf-68f7-483c-a2ac-529e9b999890
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))

# ╔═╡ 8c56595e-c1c7-47d7-b2b1-8060d0a8c778
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end

# ╔═╡ 3fea6b09-22ce-4415-9710-4ecdb31deeb0
md"""
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
"""

# ╔═╡ f4d4ceae-8818-4510-917c-2b5e9ec2af50
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

# ╔═╡ 7b1303de-eb14-45df-800a-29db77d477fd
@code_lam train(W0, b0, ones, step, steps)

# ╔═╡ f2557112-ae2f-4766-8bca-c2bf7288d1e2
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

# ╔═╡ fb8f64d3-cfb3-45d9-bd4b-92f26d16dad6
begin
    Random.seed!(6)
    B, p, k, nsteps = 256, 1024, 256, 20
    Wtrue = randn(p, k) / sqrt(p)
    Xs = [randn(B, p) for _ in 1:nsteps]
    Ys = [X * Wtrue + 0.1 * randn(B, k) for X in Xs]
    bind!(train; batch_x = i -> Xs[i+1], batch_y = i -> Ys[i+1])
    W0, b0, onesB, lr_step = zeros(p, k), zeros(k), ones(B), 2 * 0.01 / B
end;

# ╔═╡ 159325de-d5d8-498c-a10b-d6cfb77a3190
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end

# ╔═╡ a10c5d88-2056-4c9d-9208-2985dbe584e5
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step))

# ╔═╡ 38f56206-4169-443e-bc5c-3c0d344e675c
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end

# ╔═╡ 7582f65b-cac3-4371-bd59-1d9085d4ae53
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

# ╔═╡ 44756bc9-b7e6-4bb7-873d-7e0fb06ed7dd
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ 47a54bf7-050f-4904-a81a-b01dbe84442e
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ de4b7fd0-eb15-48cc-b733-70ba80f058fc
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ eeb9a866-f2e2-4622-a1d2-bcca55035702
β = gls(X, M, y)

# ╔═╡ 15dbca1f-4b4d-4830-862a-e54696bb3c00
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ 72b65f45-847d-46c7-be7e-c3641ce40995
@code_lam gls(X, M, y)

# ╔═╡ fc515119-7896-4db3-bedc-835949145ab0
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

# ╔═╡ 3fc46cdd-3439-4ec8-8a11-d942b7c3f21d
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ 171315ce-1034-4aff-9765-a5b128806047
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ b94622cb-dab8-49a6-883a-5c7a56f02ac0
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ 26052507-6aee-4e26-9d5f-53176560c872
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ 00c34f47-1c7c-4bfb-a7e0-94ca34432a39
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ 43fe437f-dc99-4257-97c2-c797f5a425ac
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 4c888186-26f5-44f9-8d9a-57e518315d59
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ cadd4991-26db-47b6-ac82-5ecc371cf309
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ b14d592d-bf6c-40de-9556-411d35f0fd73
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

# ╔═╡ 14c3828b-010f-4834-968c-407ff1e21d3b
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ 2f9b1b6c-79ef-485c-a3ad-fbd5bdfdbb52
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ c9e6b11a-ec5f-4e1d-8560-a1f3d4057225
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
"""

# ╔═╡ 6a4e6eb7-52bf-40f1-aae6-c7822071c7bc
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ bc550d0f-9ff1-42cf-bd09-28385f1bf51a
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ 4a2dec35-351d-4940-9de0-329bd993d1ed
@code_lam attention(Q, K, V)

# ╔═╡ ce60344d-3917-42f1-af29-9f7bac5417be
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ a5093011-8561-413d-8ab5-cdd947209ff1
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ 3413493a-c769-435b-af00-1b0bb84bea5d
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ 4eedae8e-522d-414f-88c3-11000a71e9e0
BLAS.get_config()

# ╔═╡ 0e2e8066-b76c-49b7-a2c4-d6dfa68d4319
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

# ╔═╡ 923e7e0c-b53f-47bd-95a6-32a040b848b5
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ 73b8e217-266a-4031-9aa3-d971615176fd
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

# ╔═╡ 0a62ea64-7dbd-4788-a0b2-fe189be7e8a9
md"""
## Appendix: the timing helper
"""

# ╔═╡ ea8601f7-8ffd-4916-be50-561806872d0c
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
# ╟─a0949dc6-2754-4ce3-ae1e-f97f8fe4aa58
# ╠═1d316dc3-c5e5-4418-bd2b-44d4e84c9af8
# ╟─0fd7bccd-e170-45b8-9989-53dc27c24eb9
# ╠═9067af62-8c37-4d1d-aa0b-746dd87390d2
# ╠═6942d067-acbb-4b3b-9a59-6c7f7854be60
# ╟─d472c25a-0785-48c6-96e1-2d2f06936cbe
# ╠═a86a0e3c-75ca-47c9-83eb-a36f151211d9
# ╠═090386b3-cffa-41e4-a237-6ead5ba90f18
# ╠═f155547f-b2ba-46b7-94c5-664bb6f14303
# ╠═139ebbaf-68f7-483c-a2ac-529e9b999890
# ╠═8c56595e-c1c7-47d7-b2b1-8060d0a8c778
# ╟─3fea6b09-22ce-4415-9710-4ecdb31deeb0
# ╠═f4d4ceae-8818-4510-917c-2b5e9ec2af50
# ╠═7b1303de-eb14-45df-800a-29db77d477fd
# ╟─f2557112-ae2f-4766-8bca-c2bf7288d1e2
# ╠═fb8f64d3-cfb3-45d9-bd4b-92f26d16dad6
# ╠═159325de-d5d8-498c-a10b-d6cfb77a3190
# ╠═a10c5d88-2056-4c9d-9208-2985dbe584e5
# ╠═38f56206-4169-443e-bc5c-3c0d344e675c
# ╟─7582f65b-cac3-4371-bd59-1d9085d4ae53
# ╠═44756bc9-b7e6-4bb7-873d-7e0fb06ed7dd
# ╟─47a54bf7-050f-4904-a81a-b01dbe84442e
# ╠═de4b7fd0-eb15-48cc-b733-70ba80f058fc
# ╠═eeb9a866-f2e2-4622-a1d2-bcca55035702
# ╟─15dbca1f-4b4d-4830-862a-e54696bb3c00
# ╠═72b65f45-847d-46c7-be7e-c3641ce40995
# ╟─fc515119-7896-4db3-bedc-835949145ab0
# ╠═3fc46cdd-3439-4ec8-8a11-d942b7c3f21d
# ╠═171315ce-1034-4aff-9765-a5b128806047
# ╠═b94622cb-dab8-49a6-883a-5c7a56f02ac0
# ╠═26052507-6aee-4e26-9d5f-53176560c872
# ╟─00c34f47-1c7c-4bfb-a7e0-94ca34432a39
# ╠═43fe437f-dc99-4257-97c2-c797f5a425ac
# ╠═4c888186-26f5-44f9-8d9a-57e518315d59
# ╟─cadd4991-26db-47b6-ac82-5ecc371cf309
# ╠═b14d592d-bf6c-40de-9556-411d35f0fd73
# ╠═14c3828b-010f-4834-968c-407ff1e21d3b
# ╠═2f9b1b6c-79ef-485c-a3ad-fbd5bdfdbb52
# ╟─c9e6b11a-ec5f-4e1d-8560-a1f3d4057225
# ╟─6a4e6eb7-52bf-40f1-aae6-c7822071c7bc
# ╠═bc550d0f-9ff1-42cf-bd09-28385f1bf51a
# ╠═4a2dec35-351d-4940-9de0-329bd993d1ed
# ╟─ce60344d-3917-42f1-af29-9f7bac5417be
# ╠═a5093011-8561-413d-8ab5-cdd947209ff1
# ╟─3413493a-c769-435b-af00-1b0bb84bea5d
# ╠═4eedae8e-522d-414f-88c3-11000a71e9e0
# ╟─0e2e8066-b76c-49b7-a2c4-d6dfa68d4319
# ╠═923e7e0c-b53f-47bd-95a6-32a040b848b5
# ╟─73b8e217-266a-4031-9aa3-d971615176fd
# ╟─0a62ea64-7dbd-4788-a0b2-fe189be7e8a9
# ╠═ea8601f7-8ffd-4916-be50-561806872d0c
