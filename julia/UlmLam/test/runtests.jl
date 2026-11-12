# The Julia frontend of ulm.lam: julia --project=julia/UlmLam -e 'using Pkg; Pkg.test()'
using UlmLam, LinearAlgebra, Random, Test

Random.seed!(7)
rel(a, b) = norm(a - b) / norm(b)
spd(n) = (A = randn(n, n); A * A' + n * I)

@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

@lam function gls_static(X::Mat(40, 6, :full_rank), M::Mat(40, 40, :spd), y::Vec(40))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

@lam function left_spd(M::Mat(:n, :n, :spd), B::Mat(:n, :k))
    M \ B
end

@lam function left_lu(M::Mat(:n, :n), B::Mat(:n, :k))
    M \ B
end

@lam function right_lu(B::Mat(:k, :n), M::Mat(:n, :n))
    B / M
end

@lam function chain(A::Mat(:n, :k), B::Mat(:k, :m), C::Mat(:m, :p))
    A * B * C
end

@lam function two(A::Mat(:n, :n), x::Vec(:n))
    (A * x, A' * A)
end

const observe = extern(:observe, [Index], Vec(:n))
const record = extern(:record, [Index, Vec(:m)])

@lam function estimate(H::Mat(:n, :m, :full_rank), R::Mat(:n, :n, :spd), steps::Index)
    loop(steps) do i
        y = observe(i)
        record(i, inv(H' * inv(R) * H) * H' * inv(R) * y)
    end
end

@lam function accumulate(A::Mat(:n, :n), B::Mat(:n, :n), X0::Mat(:n, :n), steps::Index)
    (X,) = loop(steps, X0) do i, X
        (X + A * B,)
    end
    X
end

@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

@lam function overwrite(M::InOut(Mat(:n, :n, :spd)), b::Vec(:n))
    M \ b
end

const ALL = [gls, gls_static, left_spd, left_lu, right_lu, chain, two, estimate,
             accumulate, gp, overwrite]

for backend in (:ulmblas, :cblas)
    lib = UlmLam.compile(ALL; name="jltest_$backend", backend=backend)
    @testset "$backend" begin
        @testset "GLS" begin
            for (n, m) in ((40, 6), (100, 13), (7, 7))
                X, M, y = randn(n, m), spd(n), randn(n)
                ref = (X' * (M \ X)) \ (X' * (M \ y))
                @test rel(lib.gls(X, M, y), ref) < 1e-12
            end
            X, M, y = randn(40, 6), spd(40), randn(40)
            @test rel(lib.gls_static(X, M, y), (X' * (M \ X)) \ (X' * (M \ y))) < 1e-12
            # the same kernels as from Python: Cholesky, syrk, no LU
            src = cpp(lib.gls)
            @test occursin("potrf", src) && occursin("rk<", src) && !occursin("getrf", src)
        end
        @testset "solves" begin
            M, A, B, C = spd(30), randn(30, 30), randn(30, 4), randn(4, 30)
            @test rel(lib.left_spd(M, B), M \ B) < 1e-12
            @test rel(lib.left_lu(A, B), A \ B) < 1e-10
            @test rel(lib.right_lu(C, A), C / A) < 1e-10
        end
        @testset "chain chosen at run time" begin
            for (n, k, m, p) in ((60, 3, 60, 3), (3, 60, 3, 60))
                A, B, C = randn(n, k), randn(k, m), randn(m, p)
                @test rel(lib.chain(A, B, C), A * B * C) < 1e-13
            end
        end
        @testset "two results" begin
            A, x = randn(9, 9), randn(9)
            r1, r2 = lib.two(A, x)
            @test r1 isa Vector && rel(r1, A * x) < 1e-14
            @test rel(r2, A' * A) < 1e-14
        end
        @testset "loop with callbacks" begin
            n, m, steps = 50, 5, 12
            H, R = randn(n, m), spd(n)
            ys = [randn(n) for _ in 1:steps]
            got = Vector{Vector{Float64}}(undef, steps)
            bind!(lib; observe=i -> ys[i+1], record=(i, b) -> (got[i+1] = copy(b)))
            lib.estimate(H, R, steps)
            G = (H' * (R \ H)) \ (H' / R)
            @test all(rel(got[i], G * ys[i]) < 1e-11 for i in 1:steps)
        end
        @testset "loop-carried matrix" begin
            A, B, X0 = randn(8, 8), randn(8, 8), randn(8, 8)
            @test rel(lib.accumulate(A, B, X0, 5), X0 + 5 * A * B) < 1e-13
        end
        @testset "GP with s2 * I" begin
            n, m = 60, 7
            x = randn(n + m, 2)
            Kall = [exp(-sum(abs2, x[i, :] - x[j, :])) for i in 1:n+m, j in 1:n+m]
            K, Ks, Kss, y = Kall[1:n, 1:n], Kall[1:n, n+1:end], Kall[n+1:end, n+1:end], randn(n)
            mean, cov = lib.gp(K, 0.01, Ks, Kss, y)
            Kn = K + 0.01I
            @test rel(mean, Ks' * (Kn \ y)) < 1e-10
            @test rel(cov, Kss - Ks' * (Kn \ Ks)) < 1e-10
            err = try lib.gp(K, -5.0, Ks, Kss, y); nothing catch e; e end
            @test err isa InfoError && occursin("computed from", err.meaning)
        end
        @testset "InOut and workspace" begin
            M, b = spd(20), randn(20)
            ref = M \ b
            M2 = copy(M)
            @test rel(lib.overwrite(M2, b), ref) < 1e-12
            w = Workspace(workspace_size(lib.left_spd, M, randn(20, 3)))
            B = randn(20, 3)
            @test rel(lib.left_spd(M, B; work=w), M \ B) < 1e-12
        end
        @testset "errors" begin
            @test_throws LamError lib.gls(randn(5, 3), spd(6), randn(5))
            @test_throws LamError lib.overwrite(Symmetric(spd(4)), randn(4))
        end
    end
end

@testset "frontend" begin
    @test_throws LamError Mat(3, 3, :spdd)
    @lam bad(A::Mat(3, 4), B::Mat(3, 4)) = A * B
    @test_throws LamError UlmLam.lam_module([bad])
end
