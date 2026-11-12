# GP (Gaussian process) prediction in Julia: the textbook formula
#
#     mean = K*' (K + s2 I)^-1 y,    cov = K** - K*' (K + s2 I)^-1 K*
#
# with ulm.lam, compared with what a Julia programmer writes by hand
# (cholesky, \, C.L \ K*, V'V) and with the formula written literally
# (inv).  Kernel matrices of random points (RBF kernel), m = 1000 test
# points; the times are without computing the kernel matrices.
#
#     julia --project=julia/UlmLam julia/examples/gp.jl

using UlmLam, LinearAlgebra, Random

@lam function gp(K::Mat(:n, :n, :symmetric), s2::Float, Ks::Mat(:n, :m),
                 Kss::Mat(:m, :m, :symmetric), y::Vec(:n))
    Kn = assume(K + s2 * I, :spd)
    (Ks' * inv(Kn) * y, Kss - Ks' * inv(Kn) * Ks)
end

function by_hand(K, s2, Ks, Kss, y)
    C = cholesky(Symmetric(K + s2 * I))
    V = C.L \ Ks
    (Ks' * (C \ y), Kss - V' * V)
end

literal(K, s2, Ks, Kss, y) = (Kn = inv(K + s2 * I); (Ks' * Kn * y, Kss - Ks' * Kn * Ks))

besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function rbf(A, B, ell)
    d2 = sum(abs2, A; dims=2) .+ sum(abs2, B; dims=2)' .- 2 .* (A * B')
    exp.(-max.(d2, 0) ./ (2ell^2))
end

function main(sizes=(2000, 4000, 8000), m=1000)
    lib = UlmLam.compile(gp; name="gp_jl")
    lib_ob = UlmLam.compile(gp; name="gp_jl_cblas", backend=:cblas)
    Random.seed!(1)
    s2 = 0.01
    println("m = $m test points, Julia $(VERSION) ($(basename(BLAS.get_config().loaded_libs[1].libname)), " *
            "$(BLAS.get_num_threads()) threads); seconds")
    println("     n   ulm.lam/ulmBLAS   ulm.lam/OpenBLAS   Julia by hand   Julia literal   difference")
    ok = true
    for n in sizes
        X, Xs = randn(n, 4), randn(m, 4)
        K, Ks, Kss, y = rbf(X, X, 1.5), rbf(X, Xs, 1.5), rbf(Xs, Xs, 1.5), randn(n)
        reps = n <= 4000 ? 3 : 1
        t_lam = besttime(() -> lib.gp(K, s2, Ks, Kss, y), reps)
        t_ob = besttime(() -> lib_ob.gp(K, s2, Ks, Kss, y), reps)
        t_hand = besttime(() -> by_hand(K, s2, Ks, Kss, y), reps)
        t_lit = besttime(() -> literal(K, s2, Ks, Kss, y), 1)
        (mu, S), (rmu, rS) = lib.gp(K, s2, Ks, Kss, y), by_hand(K, s2, Ks, Kss, y)
        err = max(norm(mu - rmu) / norm(rmu), norm(S - rS) / norm(rS))
        ok &= err < 1e-8
        f(t) = lpad(string(round(t, digits=3)), 7) * " (" * lpad(string(round(t / t_lam, digits=1)), 3) * "x)"
        println(lpad(n, 6), "  ", lpad(string(round(t_lam, digits=3)), 15), "   ", f(t_ob), "   ",
                f(t_hand), "  ", f(t_lit), "   ", round(err, sigdigits=2))
    end
    println(ok ? "PASS" : "FAIL")
    ok
end

main() || exit(1)
