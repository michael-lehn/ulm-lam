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

# ╔═╡ 3cf30fed-aeaf-44ba-90c4-11e13c165e70
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

# ╔═╡ bfc58bba-e510-4df9-977a-1b2f70d7a95a
begin
    import Pkg
    # the notebook's environment (see the README; on Binder set by the image)
    Pkg.activate(get(ENV, "ULM_LAM_NOTEBOOK_ENV", @__DIR__))
    using UlmLAM, LinearAlgebra, BenchmarkTools, Random, Printf, Markdown
    using AbstractGPs, KernelFunctions, PlutoUI
end

# ╔═╡ 99c79bc4-d12f-408b-bce1-bb791912fa23
md"""
## A Kalman filter, as in the textbook

`@lam` defines a ulm.lam function.  The types of the parameters carry their
sizes (`:n` and `:m` are known only at run time) and what you know about them
(`:spsd`, `:spd`: symmetric positive (semi)definite).  `loop` carries the state
``x, P`` from step to step; `extern` declares Julia functions the loop calls:
here they deliver the measurements and receive the estimates.
"""

# ╔═╡ 97d4ad9d-ce0e-4043-88c7-227cadf785f8
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

# ╔═╡ b5ea53f5-9d15-4fbe-a36e-76ce30953796
@code_lam kalman(F, Q, H, R, x0, P0, steps)

# ╔═╡ 0bb2a186-9820-45ee-a7f3-716b4971523b
md"""
``S^{-1}`` is never formed: ``K = P H^T S^{-1}`` is two triangular solves
with the Cholesky factor of ``S``.  ``H P`` is computed once and reused
(``P`` is symmetric, so ``P H^T = (H P)^T``), and ``K H P`` is evaluated as
``K (H P)``, which costs ``n^2 m`` instead of ``n^3`` for ``(K H) P``.
``x`` and ``P`` stay in place: no allocation inside the loop.

The model: a field on a periodic grid, transported and diffused, observed by
``m`` sensors.  Compared with the same formulas in plain Julia:
"""

# ╔═╡ ee47a312-0fd3-4e78-8ee7-8e8f814f384a
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

# ╔═╡ a0a59531-284d-433d-9e2b-fa2396e7693e
begin
    steps = 20
    F, Q, H, R, x0, P0, zs, truth = model(1000, 100, steps)
    estimates = zeros(1000, steps)
    bind!(kalman; measure = i -> zs[i+1], estimate = (i, x) -> (estimates[:, i+1] .= x))
end;

# ╔═╡ 9271c5a5-3c94-4ca3-ba5c-4838a1500770
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

# ╔═╡ 37f3abc7-72b1-4b11-a0bd-3661894dbdf7
md"""
And as an expert would write it in Julia: preallocated buffers, `mul!` in
place, the Cholesky factorisation of ``S`` instead of its inverse,
``H P`` once, ``K H P`` as ``K (H P)``:
"""

# ╔═╡ 52ae06ef-b048-43ff-b113-0431ea93aaa4
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

# ╔═╡ 27c17087-44c2-401b-a065-126c3f9d5031
compare("ulm.lam" => () -> kalman(F, Q, H, R, x0, P0, steps),
        "Julia, the formula as written" =>
            () -> kalman_literal(F, Q, H, R, x0, P0, zs),
        "Julia, written by an expert" =>
            () -> kalman_expert(F, Q, H, R, x0, P0, zs))

# ╔═╡ e8fcb0f1-15cf-4c72-b541-62b1849f817a
let (x, P) = kalman(F, Q, H, R, x0, P0, steps),
    (x2, P2) = kalman_literal(F, Q, H, R, x0, P0, zs),
    (x3, P3) = kalman_expert(F, Q, H, R, x0, P0, zs)
    (norm(x - x2) / norm(x2), norm(P - P2) / norm(P2),
     norm(x - x3) / norm(x3), norm(P - P3) / norm(P3))
end

# ╔═╡ 6447109e-14c1-4e9d-aaa5-cbf5b7ab0d82
md"""
## Training a linear layer: what PyTorch does, as formulas

A linear layer ``Y \approx X W + \mathbf{1} b^T`` (PyTorch's `nn.Linear`)
trained by mini-batch gradient descent on the mean squared error.  The forward
pass, the gradients and the update step are written as formulas; the batches
come from Julia functions:
"""

# ╔═╡ 5eb16b23-2882-4753-8813-46889fcb9d6b
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

# ╔═╡ ba85ed12-cf62-4517-96ed-4cb1183c3237
@code_lam train(W0, b0, ones, step, steps)

# ╔═╡ 4d0c317e-140f-47c3-8216-88006bf55422
md"""
Four BLAS calls per step, as an expert writes them, and no allocation: the
residual ``R = X W + \mathbf{1} b^T - Y`` is computed in the buffer into
which the callback delivered ``Y`` (`gemm` with ``\beta = -1``, then the
bias as a rank-1 `gemm`), the update ``W - \text{step}\, X^T R`` is one
`gemm` with ``\alpha = -\text{step}``, ``\beta = 1`` directly into ``W``,
and the bias update one `gemv`.  ``W`` stays in Julia's column-major layout
during the whole loop, and every transpose is a flag of a `gemm` (the
callbacks copy each batch into the function's buffers once).  From Python,
the same function is as fast as PyTorch on the same BLAS and 5× faster than
PyTorch's autograd for a small layer (`python/examples/torch_linear.py`).
"""

# ╔═╡ 1abdf8ed-e330-4ccc-b4f5-506f88a0ab94
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

# ╔═╡ 72093c4a-adae-4b91-a456-b3600779fda9
function train_julia(W, b, Xs, Ys, step)
    for (X, Y) in zip(Xs, Ys)
        R = X * W .+ b' .- Y
        W = W - step * (X' * R)
        b = b - step * vec(sum(R; dims = 1))
    end
    W, b
end

# ╔═╡ b5eeec03-e164-4a76-bafa-77addf9ef5c9
function train_expert(W, b, Xs, Ys, step)    # in place, BLAS calls only
    W, b = copy(W), copy(b)
    R, o = similar(Ys[1]), ones(size(Ys[1], 1))
    for (X, Y) in zip(Xs, Ys)
        copyto!(R, Y); mul!(R, X, W, 1.0, -1.0); mul!(R, o, b', 1.0, 1.0)
        mul!(W, X', R, -step, 1.0); mul!(b, R', o, -step, 1.0)
    end
    W, b
end

# ╔═╡ d889d772-1f3b-476e-b8cf-6a27e52c9934
compare("ulm.lam" => () -> train(W0, b0, onesB, lr_step, nsteps),
        "Julia, the formulas as written" => () -> train_julia(W0, b0, Xs, Ys, lr_step),
        "Julia, written by an expert" => () -> train_expert(W0, b0, Xs, Ys, lr_step))

# ╔═╡ 6ed036da-1e1f-45c4-b4d2-edf341df3727
let (W, b) = train(W0, b0, onesB, lr_step, nsteps),
    (W2, b2) = train_julia(W0, b0, Xs, Ys, lr_step)
    (norm(W - W2) / norm(W2), norm(b - b2) / norm(b2))
end

# ╔═╡ ae409110-ec80-4c4e-80c4-e3c5ea2daace
md"""
Machine learning mostly computes in single precision.  Called with `Float32`
arrays, a `@lam` function is compiled for `Float32` (a second library, as
Julia specialises on argument types): the same derivation on `sgemm`,
`spotrf`, …
"""

# ╔═╡ 42699d34-da60-41cd-a393-708ae746a4ca
let (W32, b32, o32) = (Float32.(W0), Float32.(b0), Float32.(onesB))
    compare("ulm.lam, Float64" => () -> (use32[] = false;
                                         train(W0, b0, onesB, lr_step, nsteps)),
            "ulm.lam, Float32" => () -> (use32[] = true;
                                         train(W32, b32, o32, Float32(lr_step), nsteps)))
end

# ╔═╡ 0c1edc97-a9e8-4a7f-a8cb-a5d12d3e6beb
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

# ╔═╡ 25d6307e-a965-41f7-9104-a237d85cc5dd
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

# ╔═╡ e4242cef-7de5-4391-bf2f-33966503a97e
md"""
Call it with ordinary Julia arrays.  The first call compiles it (about half a
second), later calls reuse the library.
"""

# ╔═╡ 646c2d8a-ae7b-4a0c-b6a2-0cf93ae29174
begin
    Random.seed!(1)
    n, m = 2000, 50
    X = randn(n, m)
    A = randn(n, n); M = A * A' + n * I
    y = randn(n)
end;

# ╔═╡ 24e65e03-0561-4c9e-af66-cbebbd079faf
β = gls(X, M, y)

# ╔═╡ 6ec6922d-1271-44ff-a939-f5f73f985369
md"""
## Why it is fast: `@code_lam`
"""

# ╔═╡ dd9de990-6eea-4d02-8121-2c6bf7758f7c
@code_lam gls(X, M, y)

# ╔═╡ a48cca86-ecbf-4fc4-b504-fcbaee8a2f85
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

# ╔═╡ b2640230-4621-4457-a0e2-af6609ae25fd
gls_literal(X, M, y) = inv(X' * inv(M) * X) * X' * inv(M) * y

# ╔═╡ f178487c-2b51-4746-82f7-fcc5fede5dcf
function gls_by_hand(X, M, y)
    C = cholesky(Symmetric(M))
    V = C.L \ X
    cholesky(Symmetric(V' * V)) \ (V' * (C.L \ y))
end

# ╔═╡ 2e20ac14-68f5-493b-b5af-03d2b50c599b
compare("ulm.lam" => () -> gls(X, M, y),
        "Julia, the formula as written" => () -> gls_literal(X, M, y),
        "Julia, by hand" => () -> gls_by_hand(X, M, y))

# ╔═╡ e2f414c9-1cdc-468d-89db-c3401cd1ab97
norm(β - gls_by_hand(X, M, y)) / norm(β)   # the same result

# ╔═╡ 4291ee80-fc56-4333-9e3f-48661300891f
md"""
## Gaussian process regression, compared with AbstractGPs.jl

The posterior mean and covariance at test points, as in the textbook
(``K``: kernel matrix of the training points, ``K_*`` between training and
test points, ``K_{**}`` of the test points, ``\sigma^2`` the noise variance,
a run-time `Float`):
"""

# ╔═╡ b8b04660-f826-410b-bdfa-de1f79386e4b
@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)    # what we know, the compiler cannot derive
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

# ╔═╡ 143ef898-b3dd-4246-96f8-b230ef74d5c0
@code_lam gp(K, s2, Ks, Kss, y)

# ╔═╡ 0c918607-545d-475f-a1d6-096b8860d6eb
md"""
One Cholesky factorisation for both results, ``\sigma^2`` added to the
diagonal in place, ``K_*^T K_n^{-1} K_*`` as ``V^T V`` with
``V = L^{-1} K_*`` (`trsm`, then `syrk` into ``K_{**}``'s copy).  This is
what [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
does by hand.  ulm.lam derives it from the formula:
"""

# ╔═╡ dcfd6f15-4aa9-4ecb-929e-d8f70e77a2dd
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

# ╔═╡ f731d8c5-b525-4283-b4c7-758dab389906
compare("ulm.lam (kernel matrices from KernelFunctions.jl)" =>
            () -> gp_ulmlam(xtrain, xtest, ytrain),
        "Julia, the formula as written" => () -> gp_literal(xtrain, xtest, ytrain),
        "AbstractGPs.jl" => () -> gp_abstractgps(xtrain, xtest, ytrain))

# ╔═╡ b4a90f44-bbc6-4d4a-8750-fdafbe516469
let (μ, Σ) = gp_ulmlam(xtrain, xtest, ytrain),
    (μ2, Σ2) = gp_abstractgps(xtrain, xtest, ytrain)
    (norm(μ - μ2) / norm(μ2), norm(Σ - Σ2) / norm(Σ2))
end

# ╔═╡ 8cf2a9a0-6dee-48f4-a2cf-7553d33c6722
md"""
As fast as the hand-written package, with half the memory: ulm.lam's
temporaries live in one workspace that is reused across calls.
"""

# ╔═╡ 8e84b4c7-17ca-40a0-aedd-6b6ab288e9f3
md"""
## Decisions at run time: linear attention

Linear attention (Katharopoulos et al., 2020) computes ``\phi(Q)\,\phi(K)^T V``.
From left to right it costs ``O(N^2 d)`` for a sequence of length ``N``; as
``\phi(Q)\,(\phi(K)^T V)`` it costs ``O(N d^2)``.  The sizes are known only at
run time, so ulm.lam generates both orders and picks the cheaper one when it
is called:
"""

# ╔═╡ 55442c2f-2032-4eb2-841b-7c0bc820ac74
@lam attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e)) = Q * K' * V

# ╔═╡ 4d105510-9cb2-4b8f-9974-424885c93ca8
@code_lam attention(Q, K, V)

# ╔═╡ 445cbbbb-6eb7-4dab-8e1b-cf8d914718a0
md"""
Sequence length ``N`` (head dimension ``d = 64``): $(@bind N Select([256, 1024, 4096, 8192]; default = 4096))
"""

# ╔═╡ fe4ff7f4-ccef-4a65-8cae-c01b1c9c643d
begin
    ϕ(x) = x > 0 ? x + 1 : exp(x)
    Qa, Ka, Va = ϕ.(randn(N, 64)), ϕ.(randn(N, 64)), randn(N, 64)
    compare("ulm.lam" => () -> attention(Qa, Ka, Va),
            "Julia, (Q * K') * V" => () -> (Qa * Ka') * Va,
            "Julia, Q * (K' * V)" => () -> Qa * (Ka' * Va))
end

# ╔═╡ e49efdee-9522-47df-9316-86d7d6c73b6d
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

# ╔═╡ e0fc682e-9c11-4451-bafd-91222cb89fa5
begin
    @lam bar!(M::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) = update!(y, M \ y)
    @lam foo!(M::InOut(Mat(:n, :n, :spd)), N::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) =
        update!(y, bar!(M, bar!(N, y)))
    @lam foo_matlab(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = foo!(M, N, y)
end

# ╔═╡ f4693f5d-630c-4c70-a015-54d7b8f3f53d
begin
    nl = 1000
    Mspd = (A_ = randn(nl, nl); A_ * A_' + nl * I)
    Nspd = (B_ = randn(nl, nl); B_ * B_' + nl * I)
    yv = randn(nl)
end;

# ╔═╡ 247855e9-8840-4377-862f-b123e56eca62
@code_lam foo!(Mspd, Nspd, yv)

# ╔═╡ bff18d9a-5683-44ae-84c4-58bd51999ddd
@code_lam bar!(Mspd, yv)

# ╔═╡ bc8b30cb-5caf-4562-a374-04e4badb6b4a
@code_lam foo_matlab(Mspd, Nspd, yv)

# ╔═╡ 05a849c9-c80d-498f-8da7-f0d3b4256566
let lib = UlmLAM.library(foo_matlab)                 # with foo! and bar!
    (bar! = workspace_size(lib.bar!, Mspd, yv),
     foo! = workspace_size(lib.foo!, Mspd, Nspd, yv),
     foo_matlab = workspace_size(lib.foo_matlab, Mspd, Nspd, yv))
end

# ╔═╡ 18b1222c-a775-417a-aa7c-b0af199f8edb
md"""
Workspace in bytes: none for `bar!` and `foo!`; the wrapper needs the copies
of ``M`` and ``N`` and nothing else.  It copies ``y`` directly into its
result, in which `foo!` then computes: `x = copy(y); foo!(copy(M), copy(N), x)`,
the wrapper one would write by hand.
"""

# ╔═╡ 4ede9e3e-a6af-4ddc-b344-966b59edd857
let M1 = copy(Mspd), N1 = copy(Nspd), y1 = copy(yv)
    foo!(M1, N1, y1)                                 # y1 is the solution now
    x = foo_matlab(Mspd, Nspd, yv)                    # arguments unchanged
    (error = norm(y1 - Mspd \ (Nspd \ yv)) / norm(y1),
     same_as_wrapper = y1 == x,
     M1_holds_cholesky_factor = triu(M1) ≈ cholesky(Symmetric(Mspd)).U)
end

# ╔═╡ a3964c43-89ae-415a-8991-a95f45088291
md"""
## Any BLAS

ulm.lam calls the BLAS and LAPACK Julia uses, through libblastrampoline:
"""

# ╔═╡ c5ba81a4-92f3-4343-8f40-fbd28f99aa52
BLAS.get_config()

# ╔═╡ 1bb5b753-20a2-4189-83d7-3b8540b3a6d2
md"""
With [UlmBLAS.jl](https://github.com/michael-lehn/UlmBLAS.jl) (`Pkg.add("UlmBLAS")`),
`using UlmBLAS; UlmBLAS.forward()` makes ulmBLAS the BLAS of Julia, and with
it of every ulm.lam function.  The generated code can also call the
header-only C++ library ulmBLAS directly (`UlmLAM.compile(f; backend=:ulmblas)`)
or any CBLAS (C interface to the BLAS) and LAPACK, e.g. MKL (Intel Math
Kernel Library) on a cluster.  The same functions are available from Python
with NumPy arrays.

## Under the hood: MLIR

`@lam` traces the Julia function into ulm.lam's own MLIR dialect: operations
of linear algebra on matrices whose types carry the properties.
"""

# ╔═╡ 46c2c1f8-7f52-4edf-9bb3-057b29c90c52
Text(first(UlmLAM.lam_module([gls])))

# ╔═╡ 84a29c56-a00e-4a64-8b6d-f7cb9022aab1
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

# ╔═╡ 919e9d45-9bf2-4bfa-94b2-79566cefa6c6
md"""
## Appendix: the timing helper
"""

# ╔═╡ cbde9e6f-c517-4a88-9770-c76436de766c
"""Time (best of several runs) and memory allocated per call, relative to
the first case.  The cases run in alternating rounds, after a warm-up of
all of them, so that none is measured in a quieter or busier moment."""
function compare(cases...; seconds = 20)
    fs = [f for (_, f) in cases]
    foreach(f -> f(), fs)
    t1 = maximum(f -> @elapsed(f()), fs)
    rounds = clamp(ceil(Int, seconds / (length(fs) * t1)), 3, 30)
    best = fill(Inf, length(fs))
    for r in 1:rounds, k in (isodd(r) ? eachindex(fs) : reverse(eachindex(fs)))
        GC.gc(false)
        best[k] = min(best[k], @elapsed fs[k]())
    end
    rows = [(name, best[k], @allocated(f())) for (k, (name, f)) in enumerate(cases)]
    t0 = rows[1][2]
    s = "| | time | relative | allocated |\n|:--|--:|--:|--:|\n"
    for (name, t, a) in rows
        s *= @sprintf("| %s | %.2f ms | %.1f× | %s |\n", name, 1e3t, t / t0,
                      Base.format_bytes(a))
    end
    Markdown.parse(s)
end

# ╔═╡ Cell order:
# ╟─3cf30fed-aeaf-44ba-90c4-11e13c165e70
# ╠═bfc58bba-e510-4df9-977a-1b2f70d7a95a
# ╟─99c79bc4-d12f-408b-bce1-bb791912fa23
# ╠═97d4ad9d-ce0e-4043-88c7-227cadf785f8
# ╠═b5ea53f5-9d15-4fbe-a36e-76ce30953796
# ╟─0bb2a186-9820-45ee-a7f3-716b4971523b
# ╠═ee47a312-0fd3-4e78-8ee7-8e8f814f384a
# ╠═a0a59531-284d-433d-9e2b-fa2396e7693e
# ╠═9271c5a5-3c94-4ca3-ba5c-4838a1500770
# ╟─37f3abc7-72b1-4b11-a0bd-3661894dbdf7
# ╠═52ae06ef-b048-43ff-b113-0431ea93aaa4
# ╠═27c17087-44c2-401b-a065-126c3f9d5031
# ╠═e8fcb0f1-15cf-4c72-b541-62b1849f817a
# ╟─6447109e-14c1-4e9d-aaa5-cbf5b7ab0d82
# ╠═5eb16b23-2882-4753-8813-46889fcb9d6b
# ╠═ba85ed12-cf62-4517-96ed-4cb1183c3237
# ╟─4d0c317e-140f-47c3-8216-88006bf55422
# ╠═1abdf8ed-e330-4ccc-b4f5-506f88a0ab94
# ╠═72093c4a-adae-4b91-a456-b3600779fda9
# ╠═b5eeec03-e164-4a76-bafa-77addf9ef5c9
# ╠═d889d772-1f3b-476e-b8cf-6a27e52c9934
# ╠═6ed036da-1e1f-45c4-b4d2-edf341df3727
# ╟─ae409110-ec80-4c4e-80c4-e3c5ea2daace
# ╠═42699d34-da60-41cd-a393-708ae746a4ca
# ╟─0c1edc97-a9e8-4a7f-a8cb-a5d12d3e6beb
# ╠═25d6307e-a965-41f7-9104-a237d85cc5dd
# ╟─e4242cef-7de5-4391-bf2f-33966503a97e
# ╠═646c2d8a-ae7b-4a0c-b6a2-0cf93ae29174
# ╠═24e65e03-0561-4c9e-af66-cbebbd079faf
# ╟─6ec6922d-1271-44ff-a939-f5f73f985369
# ╠═dd9de990-6eea-4d02-8121-2c6bf7758f7c
# ╟─a48cca86-ecbf-4fc4-b504-fcbaee8a2f85
# ╠═b2640230-4621-4457-a0e2-af6609ae25fd
# ╠═f178487c-2b51-4746-82f7-fcc5fede5dcf
# ╠═2e20ac14-68f5-493b-b5af-03d2b50c599b
# ╠═e2f414c9-1cdc-468d-89db-c3401cd1ab97
# ╟─4291ee80-fc56-4333-9e3f-48661300891f
# ╠═b8b04660-f826-410b-bdfa-de1f79386e4b
# ╠═143ef898-b3dd-4246-96f8-b230ef74d5c0
# ╟─0c918607-545d-475f-a1d6-096b8860d6eb
# ╠═dcfd6f15-4aa9-4ecb-929e-d8f70e77a2dd
# ╠═f731d8c5-b525-4283-b4c7-758dab389906
# ╠═b4a90f44-bbc6-4d4a-8750-fdafbe516469
# ╟─8cf2a9a0-6dee-48f4-a2cf-7553d33c6722
# ╟─8e84b4c7-17ca-40a0-aedd-6b6ab288e9f3
# ╠═55442c2f-2032-4eb2-841b-7c0bc820ac74
# ╠═4d105510-9cb2-4b8f-9974-424885c93ca8
# ╟─445cbbbb-6eb7-4dab-8e1b-cf8d914718a0
# ╠═fe4ff7f4-ccef-4a65-8cae-c01b1c9c643d
# ╟─e49efdee-9522-47df-9316-86d7d6c73b6d
# ╠═e0fc682e-9c11-4451-bafd-91222cb89fa5
# ╠═f4693f5d-630c-4c70-a015-54d7b8f3f53d
# ╠═247855e9-8840-4377-862f-b123e56eca62
# ╠═bff18d9a-5683-44ae-84c4-58bd51999ddd
# ╠═bc8b30cb-5caf-4562-a374-04e4badb6b4a
# ╠═05a849c9-c80d-498f-8da7-f0d3b4256566
# ╟─18b1222c-a775-417a-aa7c-b0af199f8edb
# ╠═4ede9e3e-a6af-4ddc-b344-966b59edd857
# ╟─a3964c43-89ae-415a-8991-a95f45088291
# ╠═c5ba81a4-92f3-4343-8f40-fbd28f99aa52
# ╟─1bb5b753-20a2-4189-83d7-3b8540b3a6d2
# ╠═46c2c1f8-7f52-4edf-9bb3-057b29c90c52
# ╟─84a29c56-a00e-4a64-8b6d-f7cb9022aab1
# ╟─919e9d45-9bf2-4bfa-94b2-79566cefa6c6
# ╠═cbde9e6f-c517-4a88-9770-c76436de766c
