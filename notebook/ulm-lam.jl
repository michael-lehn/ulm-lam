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

# ╔═╡ cf8f5d36-e49f-4e71-bc75-d29a5d010cc2
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
"""

# ╔═╡ 34186a44-87fb-4fc9-9737-b841a167b00d
begin
    import Pkg
    Pkg.activate(@__DIR__)   # the notebook's environment (see the README)
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 9b7b6e40-3942-445c-b623-5d25ce65fa35
md"""
## Write the formula

Generalised least squares: the best linear unbiased estimate of ``\beta`` in
``y = X\beta + \varepsilon`` with ``\operatorname{cov}(\varepsilon) = M``:

```math
\hat\beta = (X^T M^{-1} X)^{-1} X^T M^{-1} y
```

In Julia, with the properties as part of the types (`:n` and `:m` are sizes
known only at run time):
"""

# ╔═╡ 0007b426-c121-4be5-a487-411dd06ad4c2
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ 205cc544-e046-4a3f-a012-71be7b77c9ea
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ 0304cc9c-2c20-4070-a200-851faddd13f7
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ 3c6e2168-0f97-4796-a866-8bc9e04b6bab
β = gls(X, M, y)

# ╔═╡ 59ef028c-8974-4ad4-8ecc-0b369de8cd07
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ f8c9ab99-7e6e-423e-a1cd-410eefd4b751
@code_lam gls(X, M, y)

# ╔═╡ 75384a1f-26be-471b-bb82-fa35b02ec8da
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

# ╔═╡ 19d31e1e-ff06-4c4b-9f0b-82afc42b2c6f
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ 54555783-3e8b-4dc8-ba6d-d6096e921689
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ 5402f935-768c-4e03-bc3c-e8bfee7e90c5
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

# ╔═╡ 32bd761b-668e-4f17-8c28-203c23541291
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ 00738dd0-3b73-4fd3-8c0b-1dd884cfdd0b
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ 74439455-849b-4d9f-a3fd-91a972a2628e
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ a1a46c76-c497-4c4f-99ac-f11e6a548c8f
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 7de7602a-7e4d-4744-8f4c-4a19f5d7b694
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ d1a5945b-083f-4e6e-a895-a05e24da254f
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ 3f78cf7e-bcd7-4d25-89f9-d371fe9f44d4
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

# ╔═╡ df27d207-809e-4644-9393-90d1aa23efb8
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ db1cac5a-0382-4a99-908c-78a023330018
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ 71383705-747c-472d-b0f2-28c88390af18
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.

## A loop: the Kalman filter

Predict and update as in the textbook.  `loop` carries ``x`` and ``P`` from
step to step; the measurements come from a Julia function and the estimates
go to one (`extern`):
"""

# ╔═╡ 93f4bf49-06a6-4910-8c4a-e58ec6f844ab
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

# ╔═╡ bc606a0c-e4fa-4d80-a5dc-7e86ca667a96
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ 4803d565-8d52-4568-bbf4-a256274a4ff1
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.
"""

# ╔═╡ 8fba5f78-64fe-4ced-8425-4eb328824596
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

# ╔═╡ fcd411bb-dd34-4245-bced-ebc86226282f
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ 37a58da2-9436-4c1e-b46a-d3a72464d9cb
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

# ╔═╡ 25cf2473-0c3e-465c-b99b-6f5fd35c5dda
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))

# ╔═╡ 657e4623-dd89-46ce-88af-77d2fe79ae6b
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end

# ╔═╡ c8ac9d4a-cf1d-45e4-9f7c-6670d6f5e806
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ 93c954cf-f30c-4548-8d4d-a8180c099975
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ 451aa381-97e3-43b3-8236-f7313ef660a5
@code_lam attention(Q, K, V)

# ╔═╡ 8a4bb3d7-8acb-48fb-9c93-47c44f13972d
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ 21c7e163-fc3e-4728-960b-b312205e2270
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ e90a62ad-d8e3-4a65-ba59-fe9960964812
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ e2032850-d9fb-46ee-8bec-b4aa67603c1d
BLAS.get_config()

# ╔═╡ 3e268740-d5dc-402c-a4f2-b7c6d2b597c4
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

# ╔═╡ c117bb16-33cf-46e0-8ee0-84038a1c7a3e
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ bebc4150-174a-4649-8a23-1860ca899b04
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

# ╔═╡ Cell order:
# ╟─cf8f5d36-e49f-4e71-bc75-d29a5d010cc2
# ╠═34186a44-87fb-4fc9-9737-b841a167b00d
# ╟─9b7b6e40-3942-445c-b623-5d25ce65fa35
# ╠═0007b426-c121-4be5-a487-411dd06ad4c2
# ╟─205cc544-e046-4a3f-a012-71be7b77c9ea
# ╠═0304cc9c-2c20-4070-a200-851faddd13f7
# ╠═3c6e2168-0f97-4796-a866-8bc9e04b6bab
# ╟─59ef028c-8974-4ad4-8ecc-0b369de8cd07
# ╠═f8c9ab99-7e6e-423e-a1cd-410eefd4b751
# ╟─75384a1f-26be-471b-bb82-fa35b02ec8da
# ╠═19d31e1e-ff06-4c4b-9f0b-82afc42b2c6f
# ╠═54555783-3e8b-4dc8-ba6d-d6096e921689
# ╠═5402f935-768c-4e03-bc3c-e8bfee7e90c5
# ╠═32bd761b-668e-4f17-8c28-203c23541291
# ╠═00738dd0-3b73-4fd3-8c0b-1dd884cfdd0b
# ╟─74439455-849b-4d9f-a3fd-91a972a2628e
# ╠═a1a46c76-c497-4c4f-99ac-f11e6a548c8f
# ╠═7de7602a-7e4d-4744-8f4c-4a19f5d7b694
# ╟─d1a5945b-083f-4e6e-a895-a05e24da254f
# ╠═3f78cf7e-bcd7-4d25-89f9-d371fe9f44d4
# ╠═df27d207-809e-4644-9393-90d1aa23efb8
# ╠═db1cac5a-0382-4a99-908c-78a023330018
# ╟─71383705-747c-472d-b0f2-28c88390af18
# ╠═93f4bf49-06a6-4910-8c4a-e58ec6f844ab
# ╠═bc606a0c-e4fa-4d80-a5dc-7e86ca667a96
# ╟─4803d565-8d52-4568-bbf4-a256274a4ff1
# ╠═8fba5f78-64fe-4ced-8425-4eb328824596
# ╠═fcd411bb-dd34-4245-bced-ebc86226282f
# ╠═37a58da2-9436-4c1e-b46a-d3a72464d9cb
# ╠═25cf2473-0c3e-465c-b99b-6f5fd35c5dda
# ╠═657e4623-dd89-46ce-88af-77d2fe79ae6b
# ╟─c8ac9d4a-cf1d-45e4-9f7c-6670d6f5e806
# ╠═93c954cf-f30c-4548-8d4d-a8180c099975
# ╠═451aa381-97e3-43b3-8236-f7313ef660a5
# ╟─8a4bb3d7-8acb-48fb-9c93-47c44f13972d
# ╠═21c7e163-fc3e-4728-960b-b312205e2270
# ╟─e90a62ad-d8e3-4a65-ba59-fe9960964812
# ╠═e2032850-d9fb-46ee-8bec-b4aa67603c1d
# ╟─3e268740-d5dc-402c-a4f2-b7c6d2b597c4
# ╠═c117bb16-33cf-46e0-8ee0-84038a1c7a3e
# ╟─bebc4150-174a-4649-8a23-1860ca899b04
