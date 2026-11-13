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

# ╔═╡ 8d6df325-ecf5-41e2-8a7f-700000218957
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

# ╔═╡ f67e7c72-dade-42d3-914b-6c4ba9028295
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLam, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ afc12bb1-cd17-4e6a-991f-116932da2ca3
md"""
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
"""

# ╔═╡ ed808bde-41ab-4338-98c3-23a31cb86f47
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

# ╔═╡ ab069fd8-6af1-4a79-bab9-57df3315d11b
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ bab76fbd-0d7c-4867-bc14-5ec6f0c26dd9
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
"""

# ╔═╡ 382ca262-24fe-4a0f-9c61-92c8c231bd55
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

# ╔═╡ 480205ed-d646-44f6-93ba-2c57e5f83945
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ badde4b8-29b3-42c9-9f56-646d7dc4f4d1
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

# ╔═╡ dddf5cd2-e43c-40b3-bdd2-070c26a2cfa4
md"""
And as an expert would write it in Julia: preallocated buffers, `mul!` in
place, the Cholesky factorisation of ``S`` instead of its inverse,
``H P`` once, ``K H P`` as ``K (H P)``:
"""

# ╔═╡ ca247784-705e-4ccc-9dc6-dfaa20e03afb
function kalman_expert(F, Q, H, R, x, P, zs)
    x, P = copy(x), copy(P)
    xp, FP, HP, S, Kt, r = similar(x), similar(P), similar(H), similar(R),
                           similar(H), similar(zs[1])
    for z in zs
        mul!(xp, F, x)                              # predict
        mul!(FP, F, P); mul!(P, FP, F'); P .+= Q
        mul!(HP, H, P)                              # update: S = H P H' + R
        mul!(S, HP, H'); S .+= R
        C = cholesky!(Symmetric(S))
        copyto!(Kt, HP); ldiv!(C, Kt)               # K' = S⁻¹ H P
        mul!(r, H, xp); r .= z .- r                 # x = x + K (z - H x)
        mul!(x, Kt', r); x .+= xp
        mul!(P, Kt', HP, -1.0, 1.0)                 # P = P - K (H P)
    end
    x, P
end

# ╔═╡ f3ed0895-5e82-4eff-b8e6-653742e658f6
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs),
        "Julia, written by an expert" =>
            () -> kalman_expert(F, Q, H, R, x0, P0, zs))

# ╔═╡ 779e4e65-5c46-42ab-ba04-de8464fabdfa
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs),
    (x3, P3) = kalman_expert(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2),
     norm(x - x3) / norm(x3), norm(P - P3) / norm(P3))
end

# ╔═╡ 7b2c2598-c23c-48fb-be43-68c92bd6e27f
md"""
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
"""

# ╔═╡ 14eddb88-cf0a-4e69-8f86-1cb8b844c4f0
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

# ╔═╡ 080de39a-088e-431b-a8ac-ac82c4a85dee
@code_lam train(W0, b0, ones, step, steps)

# ╔═╡ ca7d7354-4217-4587-933b-f938d150f1da
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

# ╔═╡ 46e5161b-c768-4d6d-82f2-f5160a090018
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

# ╔═╡ d9dabf33-39e7-431c-a36f-c41140ab68ed
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end

# ╔═╡ 7881da97-3a37-4304-a06a-8786db1405e7
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step))

# ╔═╡ a0297d8f-b4c1-4c9e-a334-1a7f5a67cf99
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end

# ╔═╡ e8a0831a-9b20-44e2-a20a-f63d55b3a62d
md"""
Machine learning mostly computes in single precision.  Called with `Float32`
arrays, a `@lam` function is compiled for `Float32` (a second library, as
Julia specialises on argument types): the same derivation on `sgemm`,
`spotrf`, …
"""

# ╔═╡ 599b6659-2806-49a5-ae44-002ea2d44510
let (W32, b32, o32) = (Float32.(W0), Float32.(b0), Float32.(onesB))
    compare("ulm.lam, Float64" => () -> (use32[] = false;
                                         train(W0, b0, onesB, lr_step, nsteps)),
            "ulm.lam, Float32" => () -> (use32[] = true;
                                         train(W32, b32, o32, Float32(lr_step), nsteps)))
end

# ╔═╡ 4823e06c-bb69-4f3d-900c-9bfa840fbfc9
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

# ╔═╡ 182aed16-fde0-411a-b97d-5fdd1d4f313f
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ 89a7c40c-d912-4ed9-a660-b1640500da4e
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ 7b7acc5d-5c16-4523-910b-e33886e0cde6
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ a9e945cb-6d0a-4fe4-9f77-4c6e1d7c7693
β = gls(X, M, y)

# ╔═╡ 310b5070-b3dc-4e24-82b7-75d4c3e0bc0e
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ 76286eda-efb0-48e4-85fc-ecd764ba8aa5
@code_lam gls(X, M, y)

# ╔═╡ c96cc4d7-1246-46c1-804c-3e8b72362949
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

# ╔═╡ 6c6362a1-9dc0-44cd-81b5-02bfdb4d4ae8
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ 52125663-2228-42dd-9cc7-bd18b2654cab
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ 5a01a985-4d3d-467c-92b4-53cb7d443ff0
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ 2856c538-0472-461e-aa24-8a692de2ae0d
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ 48c1bbce-31c8-4e9a-866a-8b546f1b9f15
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ 00d2282f-ba43-4fcb-b62e-977cc52f18b5
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 2c54e334-69c0-4d5c-9126-aa48c217903d
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ eed8a2e6-4e28-44c5-b8be-2ab2b253995a
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ a2fd6b81-4e25-4eb1-a058-af7b3c63c63f
begin
    kernel = SqExponentialKernel()
    s2 = 0.01
    xtrain = 10 .* rand(4000); xtest = 10 .* rand(1000)
    ytrain = sin.(xtrain) .+ 0.1 .* randn(4000)
    gp_ulmlam(x, xs, y) =
        gp(kernelmatrix(kernel, x), s2, kernelmatrix(kernel, x, xs),
           kernelmatrix(kernel, xs), y)
    gp_abstractgps(x, xs, y) = mean_and_cov(posterior(GP(kernel)(x, s2), y)(xs))
    function gp_literal(x, xs, y)               # the formula as written
        K, Ks, Kss = kernelmatrix(kernel, x), kernelmatrix(kernel, x, xs),
                     kernelmatrix(kernel, xs)
        Kn = K + s2 * I
        (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
    end
end;

# ╔═╡ 38950547-f052-44e3-8e15-9499e790d349
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "Julia, the formula as written" => () -> gp_literal(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ e40ff418-e5c9-4a8a-8ead-1181d48fbe10
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ 4078e2c0-ae5f-47bd-8667-9a07831690c6
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
"""

# ╔═╡ 761353e4-a465-4251-84bd-30e25c52f752
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ 96ddd065-460a-4f5c-bf18-ba8c13996b02
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ fe7599d2-5b37-422a-be81-cc6ed2c7354c
@code_lam attention(Q, K, V)

# ╔═╡ 4c36482b-425a-46ff-ad40-5ce25970cf30
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ ad504986-ddec-41c5-b76f-53d4412f6fa5
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ 08ce108b-baa8-4e6e-a828-5a284ae8cd73
md"""
(Julia's `*(Q, K', V)` with three arguments also picks the order; ulm.lam
does it for any chain inside a larger expression, and together with the
rest: transposes, symmetry, solves, scalars.)

## LAPACK style, and a MATLAB-style wrapper

Functions that call functions are best written as LAPACK writes them: a
matrix that may be overwritten is `InOut`, a result goes into a parameter
(`update!`), and then no workspace is needed.  Here `bar!` overwrites ``y``
with ``M^{-1} y`` and ``M`` with its Cholesky factor (as `potrf` and `potrs`
do), and `foo!` calls it twice: ``y \leftarrow M^{-1} (N^{-1} y)``.  Who
wants it convenient writes a wrapper with read-only parameters, MATLAB
style; the copies, and the workspace for them, are then in the wrapper only.
"""

# ╔═╡ a4353286-3b9f-43eb-8665-a8a8606598a8
begin
    @lam bar!(M::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) = update!(y, M \ y)
    @lam foo!(M::InOut(Mat(:n, :n, :spd)), N::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) =
        update!(y, bar!(M, bar!(N, y)))
    @lam foo_matlab(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = foo!(M, N, y)
end

# ╔═╡ 044cb05d-d036-4136-8c0a-7c8cf511355f
begin
    nl = 1000
    Mspd = (A_ = randn(nl, nl); A_ * A_' + nl * I)
    Nspd = (B_ = randn(nl, nl); B_ * B_' + nl * I)
    yv = randn(nl)
end;

# ╔═╡ b6001ca6-a156-4185-a51e-8d2e9cd799df
@code_lam foo!(Mspd, Nspd, yv)

# ╔═╡ adc5311b-766c-474b-b7b1-c819c6e725ca
@code_lam bar!(Mspd, yv)

# ╔═╡ 5fc0d35c-15dc-4318-a202-0c8bdd95293c
@code_lam foo_matlab(Mspd, Nspd, yv)

# ╔═╡ d0658766-7c92-472d-a25a-f59468e81d5d
let lib = UlmLam.library(foo_matlab)                 # with foo! and bar!
    (bar! = workspace_size(lib.bar!, Mspd, yv),
     foo! = workspace_size(lib.foo!, Mspd, Nspd, yv),
     foo_matlab = workspace_size(lib.foo_matlab, Mspd, Nspd, yv))
end

# ╔═╡ 7ac7a998-f36d-47de-a34c-978b808db27f
md"""
Workspace in bytes: none for `bar!` and `foo!`; the wrapper needs the copies
of ``M`` and ``N`` and nothing else.  It copies ``y`` directly into its
result, in which `foo!` then computes: `x = copy(y); foo!(copy(M), copy(N), x)`,
the wrapper one would write by hand.
"""

# ╔═╡ 0a6aab80-4706-4d8a-a666-2407c6d16c1a
let M1 = copy(Mspd), N1 = copy(Nspd), y1 = copy(yv)
    foo!(M1, N1, y1)                                 # y1 is the solution now
    x = foo_matlab(Mspd, Nspd, yv)                    # arguments unchanged
    (error = norm(y1 - Mspd \ (Nspd \ yv)) / norm(y1),
     same_as_wrapper = y1 == x,
     M1_holds_cholesky_factor = triu(M1) ≈ cholesky(Symmetric(Mspd)).U)
end

# ╔═╡ 7406cc48-a4d5-49bf-a6c9-c34f90c7f459
md"""
## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ e52429b0-c52f-4e04-9621-eec62790e6a8
BLAS.get_config()

# ╔═╡ 1c08a414-e2ac-4727-8e87-8587531386fe
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

# ╔═╡ 50a97d09-9d26-443c-aefe-31e5ec8befe4
Text(first(UlmLam.lam_module([gls])))

# ╔═╡ ea900423-820d-4463-b126-926fa8f703cb
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
| in place | `InOut(Mat(...))`: a parameter the function may overwrite; `update!(p, value)`: it holds `value` on exit |
| inspect | `@code_lam f(args...)` |

In Python the same with `@lam.function`, `Matrix("n", "m", "spd")`, `@`,
`.T`, `.inv`, `lam.eye`, `lam.assume`, `lam.loop`, `lam.extern`.

So far: real double precision, dense matrices.
"""

# ╔═╡ 39a5af51-d26a-4da3-b612-f4acb8cb6087
md"""
## Appendix: the timing helper
"""

# ╔═╡ 34744aa8-cc93-455b-987a-161c29e4dce4
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
# ╟─8d6df325-ecf5-41e2-8a7f-700000218957
# ╠═f67e7c72-dade-42d3-914b-6c4ba9028295
# ╟─afc12bb1-cd17-4e6a-991f-116932da2ca3
# ╠═ed808bde-41ab-4338-98c3-23a31cb86f47
# ╠═ab069fd8-6af1-4a79-bab9-57df3315d11b
# ╟─bab76fbd-0d7c-4867-bc14-5ec6f0c26dd9
# ╠═382ca262-24fe-4a0f-9c61-92c8c231bd55
# ╠═480205ed-d646-44f6-93ba-2c57e5f83945
# ╠═badde4b8-29b3-42c9-9f56-646d7dc4f4d1
# ╟─dddf5cd2-e43c-40b3-bdd2-070c26a2cfa4
# ╠═ca247784-705e-4ccc-9dc6-dfaa20e03afb
# ╠═f3ed0895-5e82-4eff-b8e6-653742e658f6
# ╠═779e4e65-5c46-42ab-ba04-de8464fabdfa
# ╟─7b2c2598-c23c-48fb-be43-68c92bd6e27f
# ╠═14eddb88-cf0a-4e69-8f86-1cb8b844c4f0
# ╠═080de39a-088e-431b-a8ac-ac82c4a85dee
# ╟─ca7d7354-4217-4587-933b-f938d150f1da
# ╠═46e5161b-c768-4d6d-82f2-f5160a090018
# ╠═d9dabf33-39e7-431c-a36f-c41140ab68ed
# ╠═7881da97-3a37-4304-a06a-8786db1405e7
# ╠═a0297d8f-b4c1-4c9e-a334-1a7f5a67cf99
# ╟─e8a0831a-9b20-44e2-a20a-f63d55b3a62d
# ╠═599b6659-2806-49a5-ae44-002ea2d44510
# ╟─4823e06c-bb69-4f3d-900c-9bfa840fbfc9
# ╠═182aed16-fde0-411a-b97d-5fdd1d4f313f
# ╟─89a7c40c-d912-4ed9-a660-b1640500da4e
# ╠═7b7acc5d-5c16-4523-910b-e33886e0cde6
# ╠═a9e945cb-6d0a-4fe4-9f77-4c6e1d7c7693
# ╟─310b5070-b3dc-4e24-82b7-75d4c3e0bc0e
# ╠═76286eda-efb0-48e4-85fc-ecd764ba8aa5
# ╟─c96cc4d7-1246-46c1-804c-3e8b72362949
# ╠═6c6362a1-9dc0-44cd-81b5-02bfdb4d4ae8
# ╠═52125663-2228-42dd-9cc7-bd18b2654cab
# ╠═5a01a985-4d3d-467c-92b4-53cb7d443ff0
# ╠═2856c538-0472-461e-aa24-8a692de2ae0d
# ╟─48c1bbce-31c8-4e9a-866a-8b546f1b9f15
# ╠═00d2282f-ba43-4fcb-b62e-977cc52f18b5
# ╠═2c54e334-69c0-4d5c-9126-aa48c217903d
# ╟─eed8a2e6-4e28-44c5-b8be-2ab2b253995a
# ╠═a2fd6b81-4e25-4eb1-a058-af7b3c63c63f
# ╠═38950547-f052-44e3-8e15-9499e790d349
# ╠═e40ff418-e5c9-4a8a-8ead-1181d48fbe10
# ╟─4078e2c0-ae5f-47bd-8667-9a07831690c6
# ╟─761353e4-a465-4251-84bd-30e25c52f752
# ╠═96ddd065-460a-4f5c-bf18-ba8c13996b02
# ╠═fe7599d2-5b37-422a-be81-cc6ed2c7354c
# ╟─4c36482b-425a-46ff-ad40-5ce25970cf30
# ╠═ad504986-ddec-41c5-b76f-53d4412f6fa5
# ╟─08ce108b-baa8-4e6e-a828-5a284ae8cd73
# ╠═a4353286-3b9f-43eb-8665-a8a8606598a8
# ╠═044cb05d-d036-4136-8c0a-7c8cf511355f
# ╠═b6001ca6-a156-4185-a51e-8d2e9cd799df
# ╠═adc5311b-766c-474b-b7b1-c819c6e725ca
# ╠═5fc0d35c-15dc-4318-a202-0c8bdd95293c
# ╠═d0658766-7c92-472d-a25a-f59468e81d5d
# ╟─7ac7a998-f36d-47de-a34c-978b808db27f
# ╠═0a6aab80-4706-4d8a-a666-2407c6d16c1a
# ╟─7406cc48-a4d5-49bf-a6c9-c34f90c7f459
# ╠═e52429b0-c52f-4e04-9621-eec62790e6a8
# ╟─1c08a414-e2ac-4727-8e87-8587531386fe
# ╠═50a97d09-9d26-443c-aefe-31e5ec8befe4
# ╟─ea900423-820d-4463-b126-926fa8f703cb
# ╟─39a5af51-d26a-4da3-b612-f4acb8cb6087
# ╠═34744aa8-cc93-455b-987a-161c29e4dce4
