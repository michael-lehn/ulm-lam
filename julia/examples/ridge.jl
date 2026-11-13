# Ridge regression in Julia: (X'X + lambda I)^-1 X'Y, on random Fourier
# features of the bike sharing data (python/examples/ridge.py in Python).
# Compared with Julia as written (inv), Julia by hand (\, and a Cholesky
# factorisation of Symmetric(X'X + lambda I)).
#
#     julia --project=julia/UlmLAM julia/examples/ridge.jl

using UlmLAM, LinearAlgebra, Random

@lam function ridge(X::Mat(:n, :p), Y::Mat(:n, :k), lam_::Float)
    A = assume(X' * X + lam_ * I, :spd)
    inv(A) * X' * Y
end

literal(X, Y, l) = inv(X' * X + l * I) * X' * Y
by_hand(X, Y, l) = cholesky(Symmetric(X' * X + l * I)) \ (X' * Y)

include("data.jl")

besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function main()
    lib = UlmLAM.compile(ridge; name="ridge_jl", backend=:ulmblas)
    X0, y0 = bike_data()
    Random.seed!(2)
    perm = randperm(length(y0))
    test, train = perm[1:1379], perm[1380:end]
    l = 1.0
    println("Bike sharing: $(length(train)) training hours, random Fourier features, lambda = 1")
    println("Julia $(VERSION), $(BLAS.get_num_threads()) BLAS threads; seconds, (x): relative to ulm.lam")
    println("     p   RMSE   ulm.lam   Julia literal   Julia by hand   difference")
    ok = true
    for p in (500, 1000, 2000, 4000)
        W = randn(size(X0, 2), p) ./ 2
        b = 2pi .* rand(p)
        Z = sqrt(2 / p) .* cos.(X0 * W .+ b')
        X, Y = Z[train, :], reshape(y0[train], :, 1)
        reps = p <= 2000 ? 3 : 1
        t_lam = besttime(() -> lib.ridge(X, Y, l), reps)
        t_lit = besttime(() -> literal(X, Y, l), 1)
        t_hand = besttime(() -> by_hand(X, Y, l), reps)
        beta, ref = lib.ridge(X, Y, l), by_hand(X, Y, l)
        err = maximum(abs, beta - ref) / maximum(abs, ref)
        ok &= err < 1e-8
        rmse = sqrt(sum(abs2, Z[test, :] * beta[:, 1] - y0[test]) / length(test))
        f(t) = lpad(round(t, digits=3), 7) * " (" * lpad(round(t / t_lam, digits=1), 3) * "x)"
        println(lpad(p, 6), "  ", round(rmse, digits=3), "  ", lpad(round(t_lam, digits=3), 7),
                "   ", f(t_lit), "   ", f(t_hand), "   ", round(err, sigdigits=2))
    end
    println(ok ? "PASS" : "FAIL")
    ok
end

main() || exit(1)
