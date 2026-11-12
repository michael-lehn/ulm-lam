# The Kalman measurement update in a loop, in Julia (python/examples/kalman.py
# in Python):
#
#     b_i = (H' R^-1 H)^-1 H' R^-1 y_i,   i = 1, ..., steps
#
# written as on paper; ulm.lam compiles it to C++ calling ulmBLAS
# (factorisations before the loop; with enough steps the whole operator is
# formed before the loop, decided at run time).  Measurements come from a
# Julia function, the estimates go to one (callbacks).  Compared with Julia
# (LinearAlgebra, OpenBLAS): the formula as written, and the operator
# G = (H' R^-1 H)^-1 H' R^-1 computed by hand before the loop.
#
#     julia --project=julia/UlmLam julia/examples/kalman.jl

using UlmLam, LinearAlgebra, Random

const observe = extern(:observe, [Index], Vec(:n))
const record = extern(:record, [Index, Vec(:m)])

@lam function estimate(H::Mat(:n, :m, :full_rank), R::Mat(:n, :n, :spd), steps::Index)
    loop(steps) do i
        y = observe(i)
        record(i, inv(H' * inv(R) * H) * H' * inv(R) * y)
    end
end

function literal(H, R, ys)  # the formula as written, in every step
    [inv(H' * inv(R) * H) * H' * inv(R) * y for y in ys]
end

function hoisted(H, R, ys)  # by hand: G once, one product per step
    RiH = R \ H
    G = (H' * RiH) \ RiH'
    [G * y for y in ys]
end

besttime(f, reps=3) = minimum(@elapsed(f()) for _ in 1:reps)

function main(n=4000, m=200, steps=200)
    t = @elapsed lib = UlmLam.compile(estimate; name="kalman_jl")
    println("compiled in $(round(t, digits=1)) s (cached when unchanged)\n")
    println(cpp(lib.estimate))
    Random.seed!(3)
    H = rand(n, m) .* 2 .- 1
    A = rand(n, n) .* 2 .- 1
    R = A * A' / n + I
    xs = [sin.(0.1 * i .+ 0.05 .* (0:m-1)) for i in 0:steps-1]
    ys = [H * x + 0.1 * (rand(n) .* 2 .- 1) for x in xs]
    B = Vector{Vector{Float64}}(undef, steps)
    bind!(lib; observe=i -> ys[i+1], record=(i, b) -> (B[i+1] = copy(b)))
    t_lam = besttime(() -> lib.estimate(H, R, steps))
    # the literal formula takes about 2.5 s per step: 10 steps measured
    t_lit = besttime(() -> literal(H, R, ys[1:10]), 1) * steps / 10
    t_hoist = besttime(() -> hoisted(H, R, ys))
    ref = hoisted(H, R, ys)
    err = maximum(norm(B[i] - ref[i]) / norm(ref[i]) for i in 1:steps)
    println("$steps steps, n = $n, m = $m, Julia $(VERSION), $(BLAS.get_config().loaded_libs[1].libname |> basename)")
    println("  ulm.lam (generated C++, ulmBLAS)     $(round(1e3 * t_lam, digits=1)) ms")
    println("  Julia, formula as written            $(round(1e3 * t_lit, digits=1)) ms (10 steps measured)")
    println("  Julia, operator G hoisted by hand    $(round(1e3 * t_hoist, digits=1)) ms")
    println("relative difference to Julia: $(round(err, sigdigits=2))")
    println(err < 1e-10 ? "PASS" : "FAIL")
    err < 1e-10
end

main() || exit(1)
