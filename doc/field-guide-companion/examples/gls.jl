using UlmLam, LinearAlgebra

@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end

println("==== traced ulm_lam IR (frontend output)")
text, _ = UlmLam.lam_module([gls])
print(text)

X = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
M = [4.0 2.0 0.0 0.0;
     2.0 5.0 2.0 0.0;
     0.0 2.0 5.0 2.0;
     0.0 0.0 2.0 5.0]
y = [1.0, 2.0, 4.0, 4.0]

# exact reference in rational arithmetic
Xr, Mr, yr = Rational{BigInt}.(X), Rational{BigInt}.(M), Rational{BigInt}.(y)
Mi = inv(Mr)
βexact = inv(Xr' * Mi * Xr) * (Xr' * Mi * yr)
println("==== exact β (rational): ", βexact)
println("==== exact β (Float64):  ", Float64.(βexact))

β = gls(X, M, y)
println("==== ulm.lam β: ", β)
println("==== Julia expert  (X'*(M\\X)) \\ (X'*(M\\y)): ", (X' * (M \ X)) \ (X' * (M \ y)))
println("==== Julia formula inv(X'*inv(M)*X)*X'*inv(M)*y: ", inv(X' * inv(M) * X) * X' * inv(M) * y)
println("==== max abs error vs exact: ", maximum(abs.(β .- Float64.(βexact))))
println("==== isposdef(M) = ", isposdef(M), ", rank(X) = ", rank(X))
println("==== @code_lam")
display(@code_lam gls(X, M, y)); println()

println("==== failing: M symmetric but indefinite")
Mbad = copy(M); Mbad[3,3] = -1.0
try
    gls(X, Mbad, y)
catch e
    showerror(stdout, e); println()
end
println("==== failing: X rank deficient (declared full_rank)")
Xbad = [1.0 2.0; 1.0 2.0; 1.0 2.0; 1.0 2.0]
try
    println(gls(Xbad, M, y))
catch e
    showerror(stdout, e); println()
end
println("==== failing: inconsistent sizes at call time")
try
    gls(X, M, [1.0, 2.0, 3.0])
catch e
    showerror(stdout, e); println()
end
println("==== failing: non-symmetric M declared spd")
Mns = copy(M); Mns[1,2] = 100.0
try
    println(gls(X, Mns, y), "   vs expert with Mns: ", (X' * (Mns \ X)) \ (X' * (Mns \ y)), "  vs with lower triangle mirrored: ", let S = Symmetric(Mns, :L); (X' * (S \ X)) \ (X' * (S \ y)) end)
catch e
    showerror(stdout, e); println()
end
println("==== trace-time error: symbolic sizes")
@lam bad(A::Mat(:n, :m), B::Mat(:n, :m)) = A * B
try
    UlmLam.lam_module([bad])
catch e
    showerror(stdout, e); println()
end
println("==== versions: Julia ", VERSION, ", BLAS: ", BLAS.get_config())
