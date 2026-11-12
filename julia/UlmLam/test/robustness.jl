# What a user tries first: wrappers and views as arguments, other element
# types, empty matrices, calls from several threads, powers, I, and clear
# errors for what is not supported.
using UlmLam, LinearAlgebra, Random, Test

@testset "robustness" begin
    Random.seed!(11)
    @lam rb_gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n)) =
        inv(X' * inv(M) * X) * X' * inv(M) * y
    ref(X, M, y) = (X' * (M \ X)) \ (X' * (M \ y))
    X = randn(40, 3); A = randn(40, 40); M = A * A' + 40I; y = randn(40)
    rel(a, b) = norm(vec(a) - vec(b)) / norm(b)
    r = ref(X, M, y)
    @test rel(rb_gls(X, Symmetric(M), y), r) < 1e-10
    @test rel(rb_gls(view(X, :, 1:3), M, y), r) < 1e-10
    P = randn(80, 3); V = view(P, 1:2:80, :)
    @test rel(rb_gls(V, M, y), ref(Matrix(V), M, y)) < 1e-10
    @test rel(rb_gls(Float32.(X), Float32.(M), Float32.(y)), r) < 1e-4
    @test rel(rb_gls(X, Diagonal(diag(M)), y), ref(X, Matrix(Diagonal(diag(M))), y)) < 1e-10
    # empty: m = 0 (no BLAS error message, an empty result)
    @test length(rb_gls(randn(40, 0), M, y)) == 0
    # several threads at once: one workspace per task
    rs = fetch.([Threads.@spawn rb_gls(X, M, y) for _ in 1:8])
    @test all(x -> rel(x, r) < 1e-10, rs)
    # powers, I, the identity as a result
    @lam rb_pow(A::Mat(:n, :n), x::Vec(:n)) = A^2 * x + A^-1 * x
    @test rel(rb_pow(A, y), A^2 * y + A \ y) < 1e-10
    @lam rb_eye(A::Mat(:n, :n)) = A^0
    @test rb_eye(A) == Matrix(1.0I, 40, 40)
    @lam rb_i(A::Mat(:n, :n), x::Vec(:n)) = A * I * x + (2I) * x - 0 * x
    @test rel(rb_i(A, y), A * y + 2y) < 1e-12
    @lam rb_dot(x::Vec(:n), y::Vec(:n)) = dot(x, y)
    @test rb_dot(y, y)[1] ≈ dot(y, y)
    # clear errors
    @lam rb_tr(A::Mat(:n, :n)) = tr(A)
    @test_throws LamError rb_tr(A)
    @lam rb_bc(x::Vec(:n)) = x .* x
    @test_throws LamError rb_bc(y)
    @test_throws LamError rb_gls(X, M, randn(39))
    @test_throws InfoError rb_gls(X, -M, y)
end
