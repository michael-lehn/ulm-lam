# Bayesian linear regression in Julia (python/examples/bayes.py in Python):
#
#     A = alpha I + beta Phi'Phi,  w = beta A^-1 Phi't,
#     mean = Phi* w,  cov = Phi* A^-1 Phi*' + I / beta
#
# on random Fourier features of the bike sharing data, predictive mean and
# full covariance of all test hours.  Compared with Julia as written (inv)
# and by hand (cholesky, \, C.L \ Phi*', V'V).
#
#     julia --project=julia/UlmLam julia/examples/bayes.jl

using UlmLam, LinearAlgebra, Random
include("data.jl")

@lam function blr(Phi::Mat(:n, :p), t::Vec(:n), Phis::Mat(:m, :p), alpha::Float, beta::Float)
    A = assume(alpha * I + beta * (Phi' * Phi), :spd)
    w = beta * inv(A) * Phi' * t
    (Phis * w, Phis * inv(A) * Phis' + (1 / beta) * I)
end

function literal(Phi, t, Phis, alpha, beta)
    S = inv(alpha * I + beta * Phi' * Phi)
    w = beta * S * Phi' * t
    (Phis * w, Phis * S * Phis' + I / beta)
end

function by_hand(Phi, t, Phis, alpha, beta)
    C = cholesky(Symmetric(alpha * I + beta * (Phi' * Phi)))
    w = beta * (C \ (Phi' * t))
    V = C.L \ Phis'
    (Phis * w, V' * V + I / beta)
end

besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function main()
    lib = UlmLam.compile(blr; name="bayes_jl", backend=:ulmblas)
    X0, y0 = bike_data()
    Random.seed!(2)
    perm = randperm(length(y0))
    test, train = perm[1:1379], perm[1380:end]
    alpha, beta = 1.0, 10.0
    println("Bike sharing: $(length(train)) training hours, $(length(test)) test hours, Julia $(VERSION), " *
            "$(BLAS.get_num_threads()) BLAS threads; seconds, (x): relative to ulm.lam")
    println("     p   RMSE   ulm.lam   Julia as written   Julia by hand   difference")
    ok = true
    for p in (500, 1000, 2000, 4000)
        W = randn(size(X0, 2), p) ./ 2
        b = 2pi .* rand(p)
        Z = sqrt(2 / p) .* cos.(X0 * W .+ b')
        Phi, t, Phis = Z[train, :], y0[train], Z[test, :]
        reps = p <= 2000 ? 3 : 1
        t_lam = besttime(() -> lib.blr(Phi, t, Phis, alpha, beta), reps)
        t_lit = besttime(() -> literal(Phi, t, Phis, alpha, beta), 1)
        t_hand = besttime(() -> by_hand(Phi, t, Phis, alpha, beta), reps)
        (m, c), (rm, rc) = lib.blr(Phi, t, Phis, alpha, beta), by_hand(Phi, t, Phis, alpha, beta)
        err = max(maximum(abs, m - rm) / maximum(abs, rm), maximum(abs, c - rc) / maximum(abs, rc))
        ok &= err < 1e-8
        rmse = sqrt(sum(abs2, m - y0[test]) / length(test))
        f(t) = lpad(round(t, digits=3), 7) * " (" * lpad(round(t / t_lam, digits=1), 3) * "x)"
        println(lpad(p, 6), "  ", round(rmse, digits=3), "  ", lpad(round(t_lam, digits=3), 7), "   ",
                f(t_lit), "    ", f(t_hand), "   ", round(err, sigdigits=2))
    end
    println(ok ? "PASS" : "FAIL")
    ok
end

main() || exit(1)
