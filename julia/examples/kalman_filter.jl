# The Kalman filter (predict and update) in Julia, data assimilation on a
# periodic grid (python/examples/kalman_filter.py in Python).  Compared with
# Julia as written (inv(S), P - K * H * P) and by hand (H P once, Cholesky).
#
#     julia --project=julia/UlmLam julia/examples/kalman_filter.jl

using UlmLam, LinearAlgebra, Random

const measure = extern(:measure, [Index], Vec(:m))
const estimate = extern(:estimate, [Index, Vec(:n)])

@lam function kalman(F::Mat(:n, :n), Q::Mat(:n, :n, :spsd), H::Mat(:m, :n),
                     R::Mat(:m, :m, :spd), x0::Vec(:n), P0::Mat(:n, :n, :spd), steps::Index)
    loop(steps, x0, P0) do i, x, P
        P = assume(P, :symmetric)          # a covariance stays symmetric
        x = F * x                                      # predict
        P = F * P * F' + Q
        S = assume(H * P * H' + R, :spd)               # update
        K = P * H' * inv(S)
        x = x + K * (measure(i) - H * x)
        P = P - K * H * P
        estimate(i, x)
        (x, P)
    end
end

function literal(F, Q, H, R, x, P, zs)
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

function by_hand(F, Q, H, R, x, P, zs)
    for z in zs
        x = F * x
        P = F * P * F' + Q
        HP = H * P
        C = cholesky(Symmetric(HP * H' + R))
        K = (C \ HP)'
        x = x + K * (z - H * x)
        P = P - K * HP
    end
    x, P
end

function model(n, m, steps)
    c, nu = 0.4, 0.2
    F = zeros(n, n)
    for i in 1:n
        F[i, i] = 1 - 2nu
        F[i, mod1(i + 1, n)] = nu - c / 2
        F[i, mod1(i - 1, n)] = nu + c / 2
    end
    d = [min(abs(i - j), n - abs(i - j)) for i in 1:n, j in 1:n]
    Q = 0.01 .* exp.(-d ./ 10)
    sensors = sort(randperm(n)[1:m])
    H = zeros(m, n)
    for (k, s) in enumerate(sensors)
        H[k, s] = 1
    end
    R = 0.05 * Matrix(I, m, m)
    Lq = cholesky(Symmetric(Q + 1e-12I)).L
    truth = [sin(2pi * i / n) + 0.5 * sin(6pi * i / n) for i in 0:n-1]
    zs = Vector{Vector{Float64}}()
    for _ in 1:steps
        truth = F * truth + Lq * randn(n)
        push!(zs, H * truth + sqrt(0.05) * randn(m))
    end
    F, Q, H, R, zeros(n), Matrix(1.0I, n, n), zs
end

besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function main()
    lib = UlmLam.compile(kalman; name="kalman_filter_jl")
    Random.seed!(5)
    println("Kalman filter, Julia $(VERSION); seconds, (x): relative to ulm.lam")
    println("     n     m  steps   ulm.lam   Julia as written   Julia by hand   difference")
    ok = true
    for (n, m, steps) in ((500, 50, 50), (1000, 100, 50), (2000, 200, 20))
        F, Q, H, R, x0, P0, zs = model(n, m, steps)
        last = zeros(n)
        bind!(lib; measure=i -> zs[i+1], estimate=(i, x) -> (last .= x))
        reps = n <= 1000 ? 3 : 1
        t_lam = besttime(() -> lib.kalman(F, Q, H, R, x0, P0, steps), reps)
        t_lit = besttime(() -> literal(F, Q, H, R, x0, P0, zs), 1)
        t_hand = besttime(() -> by_hand(F, Q, H, R, x0, P0, zs), reps)
        xl, Pl = lib.kalman(F, Q, H, R, x0, P0, steps)
        xh, Ph = by_hand(F, Q, H, R, x0, P0, zs)
        err = max(norm(xl - xh) / norm(xh), norm(Pl - Ph) / norm(Ph))
        ok &= err < 1e-9
        f(t) = lpad(round(t, digits=3), 7) * " (" * lpad(round(t / t_lam, digits=1), 3) * "x)"
        println(lpad(n, 6), lpad(m, 6), lpad(steps, 7), "   ", lpad(round(t_lam, digits=3), 7), "   ",
                f(t_lit), "    ", f(t_hand), "   ", round(err, sigdigits=2))
    end
    println(ok ? "PASS" : "FAIL")
    ok
end

main() || exit(1)
