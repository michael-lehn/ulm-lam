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

# ╔═╡ 1f86d865-3935-454e-a4e8-cb6425df5e31
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

# ╔═╡ 6ec8deea-59ba-4a3e-afb6-95e39933c992
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 48cfa4a2-d85c-4f4f-aeca-9291e4cd1ab8
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

# ╔═╡ 4b6a0989-6240-41ab-9b86-dd7ecdfdcc8c
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ ae77e6a4-2440-4582-820b-0678d5f37730
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ c64bdd76-f0ff-4cd6-b67c-ab8c6ee871b3
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ 2ad231a6-a084-47d5-9ca3-3d8dce2c11ae
β = gls(X, M, y)

# ╔═╡ 35e3d952-dec5-4d4c-902b-e1f0d4a8099e
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ d9e919a8-a84c-4f32-9218-bb19473679eb
@code_lam gls(X, M, y)

# ╔═╡ 2ad3b06d-92a8-4935-aa48-f1f3547d52b7
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

# ╔═╡ ce228192-aa94-4d58-84b6-0d534693c894
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ ed1178e4-7e5a-4696-992a-a5c71680e60f
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ dd9a2d2a-4602-46a0-a1b7-dec2f39e6963
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

# ╔═╡ e3f544dc-3b00-4d4a-a6df-09563eddaec2
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ 33626ac4-8046-4691-8dff-497a96108edb
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ afabe02a-51a0-47aa-a80a-c6899d99f3c6
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ b2cef0aa-1713-4e80-a6d0-7c07fbee9c2b
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ f14cf7b3-e69a-479c-a81f-1eb1928862ff
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ 72916aad-2ed2-43e0-a88c-c9776278419c
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ 14ed5dc9-9f6c-43a3-913a-519d9bd224a7
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

# ╔═╡ eb4980cd-0146-4629-a21c-940ee78b6fa2
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ 7d1053fe-1de8-498d-9e72-a8c4f3c54508
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ 619544bf-3097-4aa1-ba9f-38d42a5ef98d
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.

## A loop: the Kalman filter

Predict and update as in the textbook.  `loop` carries ``x`` and ``P`` from
step to step; the measurements come from a Julia function and the estimates
go to one (`extern`):
"""

# ╔═╡ 13c30aff-eaf2-4f25-88dc-bc8b53aaf107
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

# ╔═╡ 92b661fe-eb74-42b6-a73b-4413592a4c46
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ be9de426-96ca-47db-8a84-419a567d7b03
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.
"""

# ╔═╡ 679ffed0-23f5-40fd-8144-0cc9fedb5915
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

# ╔═╡ ca3f7101-36cb-4666-a0e8-b7c10bf4e16d
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ 77b9e80c-bc61-4105-accd-506d09899067
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

# ╔═╡ bc28e36e-a76d-407f-b09e-cd1837598769
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs))

# ╔═╡ f8fbe8fd-95b1-4a81-ab4d-322b18bebacb
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2))
end

# ╔═╡ f14629d7-76f1-4520-9dfb-4d56a888b05d
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ cb3eea95-b703-44bb-8c96-3dbe825af206
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ 06c276ae-9b25-4a53-81ec-7e7b11806aa7
@code_lam attention(Q, K, V)

# ╔═╡ 5fa2be83-0c48-48bd-b396-6694f753376e
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ 5e8c8dd3-1331-4727-bb28-13846897758b
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ dd8a20ce-d848-4abb-8dc1-4d68e4528941
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ f811f80d-b4c2-460a-a9be-7af74a90354f
BLAS.get_config()

# ╔═╡ 0b70a5e3-1b6c-43aa-a63f-a35eeff8087d
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

# ╔═╡ 7f2bf9b0-a6f3-45b3-ae0e-885d52b26e04
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ b6079899-cdea-4297-8b34-031ab15a3f78
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
# ╟─1f86d865-3935-454e-a4e8-cb6425df5e31
# ╠═6ec8deea-59ba-4a3e-afb6-95e39933c992
# ╟─48cfa4a2-d85c-4f4f-aeca-9291e4cd1ab8
# ╠═4b6a0989-6240-41ab-9b86-dd7ecdfdcc8c
# ╟─ae77e6a4-2440-4582-820b-0678d5f37730
# ╠═c64bdd76-f0ff-4cd6-b67c-ab8c6ee871b3
# ╠═2ad231a6-a084-47d5-9ca3-3d8dce2c11ae
# ╟─35e3d952-dec5-4d4c-902b-e1f0d4a8099e
# ╠═d9e919a8-a84c-4f32-9218-bb19473679eb
# ╟─2ad3b06d-92a8-4935-aa48-f1f3547d52b7
# ╠═ce228192-aa94-4d58-84b6-0d534693c894
# ╠═ed1178e4-7e5a-4696-992a-a5c71680e60f
# ╠═dd9a2d2a-4602-46a0-a1b7-dec2f39e6963
# ╠═e3f544dc-3b00-4d4a-a6df-09563eddaec2
# ╠═33626ac4-8046-4691-8dff-497a96108edb
# ╟─afabe02a-51a0-47aa-a80a-c6899d99f3c6
# ╠═b2cef0aa-1713-4e80-a6d0-7c07fbee9c2b
# ╠═f14cf7b3-e69a-479c-a81f-1eb1928862ff
# ╟─72916aad-2ed2-43e0-a88c-c9776278419c
# ╠═14ed5dc9-9f6c-43a3-913a-519d9bd224a7
# ╠═eb4980cd-0146-4629-a21c-940ee78b6fa2
# ╠═7d1053fe-1de8-498d-9e72-a8c4f3c54508
# ╟─619544bf-3097-4aa1-ba9f-38d42a5ef98d
# ╠═13c30aff-eaf2-4f25-88dc-bc8b53aaf107
# ╠═92b661fe-eb74-42b6-a73b-4413592a4c46
# ╟─be9de426-96ca-47db-8a84-419a567d7b03
# ╠═679ffed0-23f5-40fd-8144-0cc9fedb5915
# ╠═ca3f7101-36cb-4666-a0e8-b7c10bf4e16d
# ╠═77b9e80c-bc61-4105-accd-506d09899067
# ╠═bc28e36e-a76d-407f-b09e-cd1837598769
# ╠═f8fbe8fd-95b1-4a81-ab4d-322b18bebacb
# ╟─f14629d7-76f1-4520-9dfb-4d56a888b05d
# ╠═cb3eea95-b703-44bb-8c96-3dbe825af206
# ╠═06c276ae-9b25-4a53-81ec-7e7b11806aa7
# ╟─5fa2be83-0c48-48bd-b396-6694f753376e
# ╠═5e8c8dd3-1331-4727-bb28-13846897758b
# ╟─dd8a20ce-d848-4abb-8dc1-4d68e4528941
# ╠═f811f80d-b4c2-460a-a9be-7af74a90354f
# ╟─0b70a5e3-1b6c-43aa-a63f-a35eeff8087d
# ╠═7f2bf9b0-a6f3-45b3-ae0e-885d52b26e04
# ╟─b6079899-cdea-4297-8b34-031ab15a3f78
