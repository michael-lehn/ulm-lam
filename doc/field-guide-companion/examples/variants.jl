using UlmLam, LinearAlgebra

@lam function gls_nofr(X::Mat(:n, :m), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
@lam function gls_nospd(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
@lam function gls_sym(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :symmetric), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
@lam function gls_assume(X::Mat(:n, :m), M::Mat(:n, :n, :spd), y::Vec(:n))
    S = assume(X' * inv(M) * X, :spd)
    inv(S) * X' * inv(M) * y
end
@lam function iterate(M::Mat(:n, :n, :spd), x0::Vec(:n), steps::Index)
    loop(steps, x0) do i, x
        (inv(M) * x,)
    end
end
@lam function shared(A::Mat(:n, :k), B::Mat(:k, :n), C::Mat(:n, :n))
    P = A * B
    (P * C, P + C)
end
X = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
M = [4.0 2.0 0.0 0.0; 2.0 5.0 2.0 0.0; 0.0 2.0 5.0 2.0; 0.0 0.0 2.0 5.0]
y = [1.0, 2.0, 4.0, 4.0]
for f in (gls_nofr, gls_nospd, gls_sym, gls_assume)
    println("==== ", f.name)
    display(UlmLam.code_lam(f)); println()
    println("result: ", f(X, M, y))
end
println("==== iterate")
display(UlmLam.code_lam(iterate)); println()
println("result: ", iterate(M, y, 3), "  reference: ", M \ (M \ (M \ y)))
println("==== shared")
A = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]; B = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]; C = Matrix(1.0I, 4, 4) .+ 1
display(UlmLam.code_lam(shared)); println()
r1, r2 = shared(A, B, C)
println("ok: ", r1 ≈ A*B*C, " ", r2 ≈ A*B + C)
println("==== Mns2 (lower triangle in Julia changed, M declared spd)")
Mns = copy(M); Mns[2,1] = 100.0
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
println("ulm.lam: ", gls(X, Mns, y))
println("Julia with Symmetric(Mns, :U): ", let S = Symmetric(Mns, :U); (X' * (S \ X)) \ (X' * (S \ y)) end)
println("Julia with Mns as given:       ", (X' * (Mns \ X)) \ (X' * (Mns \ y)))
