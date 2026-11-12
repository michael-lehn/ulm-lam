# L D L^T for symmetric indefinite matrices and least squares by QR (where
# the backend has LAPACK; ulmBLAS: LU and the normal equations).
using UlmLam, LinearAlgebra, Random, Test

@testset "LDL^T and QR" begin
    Random.seed!(21)
    n = 40
    A = randn(n, n); S = A + A'                       # symmetric, indefinite
    B = randn(n, 3)
    X = randn(120, 5); y = randn(120); Y = randn(120, 2)
    @lam fa_sym(S::Mat(:n, :n, :symmetric, :non_singular), B::Mat(:n, :k)) = S \ B
    @lam fa_symr(C::Mat(:k, :n), S::Mat(:n, :n, :symmetric, :non_singular)) = C / S
    @lam fa_ls(X::Mat(:m, :n, :full_rank), y::Vec(:m)) = X \ y
    @lam fa_lsm(X::Mat(:m, :n, :full_rank), Y::Mat(:m, :k)) = lstsq(X, Y)
    rel(a, b) = norm(a - b) / norm(b)
    t = range(0, 1, length=100); V = [ti^j for ti in t, j in 0:11]; xt = randn(12)
    for backend in (:julia, :ulmblas, :cblas)
        lib = UlmLam.compile([fa_sym, fa_symr, fa_ls, fa_lsm]; name="fact_$backend",
                             backend=backend)
        @test rel(lib.fa_sym(S, B), S \ B) < 1e-10
        @test rel(lib.fa_symr(B', S), B' / S) < 1e-10
        @test rel(lib.fa_ls(X, y), X \ y) < 1e-12
        @test rel(lib.fa_lsm(X, Y), X \ Y) < 1e-12
        sel = read(replace(lib.path, r"lib([^/]*)\.[^.]+$" => s"\1.selected.mlir"), String)
        if backend == :ulmblas   # no LAPACK: LU, the normal equations
            @test !occursin("symmetric", sel) && !occursin("gels", sel)
        else                     # LAPACK: sytrf, QR
            @test occursin("{symmetric}", sel) && occursin("ulm_lamk.gels", sel)
            # QR on an ill-conditioned least squares problem (cond 1e8)
            @test rel(lib.fa_ls(V, V * xt), xt) < 1e-6
        end
    end
    @test occursin("sytrf", string(@code_lam fa_sym(S, B)))
    @test occursin("gels", string(@code_lam fa_ls(X, y)))
end
