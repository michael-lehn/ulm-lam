# Linear attention in Julia: phi(Q) phi(K)' V, the order chosen at run time
# by ulm.lam (python/examples/attention.py in Python).  Julia's * with three
# arguments already chooses the order of a chain of three matrices (since
# Julia 1.7); written with two products, (Q * K') * V, it does not.
# ulm.lam runs on ulmBLAS (whose gemm is single threaded so far) and on
# OpenBLAS with as many threads as Julia's BLAS.
#
#     julia --project=julia/UlmLam julia/examples/attention.jl

using UlmLam, LinearAlgebra, Random
# the OpenBLAS of backend=:cblas with as many threads as Julia's BLAS
ENV["OPENBLAS_NUM_THREADS"] = string(BLAS.get_num_threads())

@lam function attention(Q::Mat(:N, :d), K::Mat(:N, :d), V::Mat(:N, :e))
    Q * K' * V
end

phi(x) = x > 0 ? x + 1 : exp(x)
besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function main()
    lib = UlmLam.compile(attention; name="attention_jl", backend=:ulmblas)
    lib_ob = UlmLam.compile(attention; name="attention_jl_cblas", backend=:cblas)
    Random.seed!(4)
    d = 64
    println("linear attention, d = $d, Julia $(VERSION), $(BLAS.get_num_threads()) BLAS threads; seconds, (x): relative to ulm.lam/OpenBLAS")
    println("      N   ulm.lam/OpenBLAS   ulm.lam/ulmBLAS   Julia *(Q, K', V)   Julia (Q * K') * V   Julia Q * (K' * V)")
    ok = true
    for N in (256, 1024, 4096, 16384)
        Q, K = phi.(randn(N, d)), phi.(randn(N, d))
        V1 = hcat(randn(N, d), ones(N))
        reps = N <= 4096 ? 5 : 2
        t_lam = besttime(() -> lib_ob.attention(Q, K, V1), reps)
        t_ulm = besttime(() -> lib.attention(Q, K, V1), reps)
        t_3 = besttime(() -> Q * K' * V1, reps)
        t_l = besttime(() -> (Q * K') * V1, N > 4096 ? 1 : reps)
        t_r = besttime(() -> Q * (K' * V1), reps)
        r, ref = lib.attention(Q, K, V1), Q * (K' * V1)
        err = maximum(abs, r[:, 1:d] ./ r[:, d+1] - ref[:, 1:d] ./ ref[:, d+1]) / maximum(abs, ref[:, 1:d] ./ ref[:, d+1])
        ok &= err < 1e-12
        f(t) = lpad(round(t, digits=4), 7) * " (" * lpad(round(t / t_lam, digits=1), 5) * "x)"
        println(lpad(N, 7), "  ", lpad(round(t_lam, digits=4), 12), "     ", f(t_ulm), "    ", f(t_3), "      ", f(t_l), "      ", f(t_r))
    end
    println(ok ? "PASS" : "FAIL")
    ok
end

main() || exit(1)
