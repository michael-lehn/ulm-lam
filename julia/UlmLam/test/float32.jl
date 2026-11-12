# Float32: a @lam function called with Float32 arrays is compiled for f32
# (all kernels are generic in the element type); results are Float32.
using UlmLam, LinearAlgebra, Random, Test

@testset "Float32" begin
    Random.seed!(12)
    @lam f32_gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n)) =
        inv(X' * inv(M) * X) * X' * inv(M) * y
    @lam f32_ops(A::Mat(:n, :n), L::Mat(:n, :n, :lower_tri), x::Vec(:n), a::Float) =
        a * (A * x) + L * x - inv(A) * x / a
    X = randn(60, 4); A0 = randn(60, 60); M = A0 * A0' + 60I; y = randn(60)
    A = randn(60, 60) + 20I; L = tril(randn(60, 60)) + 20I
    ref = (X' * (M \ X)) \ (X' * (M \ y))
    ref2 = 1.5 * (A * y) + L * y - (A \ y) / 1.5
    rel(a, b) = norm(a - b) / norm(b)
    for backend in (:julia, :ulmblas, :cblas)
        lib = UlmLam.compile([f32_gls, f32_ops]; name="f32_test_$backend", backend=backend,
                             eltype=Float32)
        r = lib.f32_gls(Float32.(X), Float32.(M), Float32.(y))
        @test eltype(r) == Float32
        @test rel(r, ref) < 1e-4
        r2 = lib.f32_ops(Float32.(A), Float32.(L), Float32.(y), 1.5f0)
        @test eltype(r2) == Float32 && rel(r2, ref2) < 1e-4
    end
    # a call with Float32 arrays chooses the Float32 library, mixed ones Float64
    @test eltype(f32_gls(Float32.(X), Float32.(M), Float32.(y))) == Float32
    @test eltype(f32_gls(Float32.(X), M, y)) == Float64
end
