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

@testset "expanded factors (Kalman update)" begin
    Random.seed!(23)
    ef_measure = extern(:ef_measure, [Index], Vec(:m))
    ef_estimate = extern(:ef_estimate, [Index, Vec(:n)])
    @lam function ef_kalman(F::Mat(:n, :n), Q::Mat(:n, :n, :spsd), H::Mat(:m, :n),
                            R::Mat(:m, :m, :spd), x0::Vec(:n), P0::Mat(:n, :n, :spd),
                            steps::Index)
        loop(steps, x0, P0) do i, x, P
            P = assume(P, :symmetric)
            x = F * x
            P = F * P * F' + Q
            S = assume(H * P * H' + R, :spd)
            K = P * H' * inv(S)
            x = x + K * (ef_measure(i) - H * x)
            P = P - K * H * P
            ef_estimate(i, x)
            (x, P)
        end
    end
    n, m, steps = 40, 6, 8
    F = randn(n, n) / sqrt(n); A = randn(n, n); Q = A * A' / n; H = randn(m, n)
    R = Matrix(1.0I, m, m); x0 = randn(n); P0 = Matrix(1.0I, n, n)
    zs = [randn(m) for _ in 1:steps]
    bind!(ef_kalman; ef_measure = i -> zs[i+1], ef_estimate = (i, x) -> nothing)
    x, P = ef_kalman(F, Q, H, R, x0, P0, steps)
    xr, Pr = x0, P0
    for z in zs
        xr = F * xr; Pr = F * Pr * F' + Q; S = H * Pr * H' + R; K = Pr * H' * inv(S)
        xr = xr + K * (z - H * xr); Pr = Pr - K * H * Pr
    end
    @test norm(x - xr) / norm(xr) < 1e-12 && norm(P - Pr) / norm(Pr) < 1e-12
    @test P == P'                     # syrk: symmetric by construction
    t = string(@code_lam ef_kalman())
    @test occursin(r"syrk +T\d+ ← T\d+ − (T\d+ T\d+ᵀ|T\d+ᵀ T\d+)", t)
    # K is never formed: one trsm (W = L⁻¹ H P), K r as Wᵀ (L⁻¹ r)
    @test count("trsm", t) == 1 && occursin("trsv", t)
    @test !occursin("if cheaper", t)
end
