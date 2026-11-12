using UlmLam, LinearAlgebra
@lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd), y::Vec(:n))
    inv(X' * inv(M) * X) * X' * inv(M) * y
end
setprecision(BigFloat, 256)
M = [4.0 2.0 0.0 0.0; 2.0 5.0 2.0 0.0; 0.0 2.0 5.0 2.0; 0.0 0.0 2.0 5.0]
y = [1.0, 2.0, 4.0, 4.0]
for ε in (1e-2, 1e-5, 1e-7, 1e-9)
    X = [1.0 1.0; 1.0 1.0+ε; 1.0 1.0+2ε; 1.0 1.0+3ε]
    Xb, Mb, yb = big.(X), big.(M), big.(y)
    ref = (Xb' * (Mb \ Xb)) \ (Xb' * (Mb \ yb))       # exact up to 256-bit
    L = cholesky(Symmetric(M)).L                         # whitened least squares by QR
    βqr = qr(L \ X) \ (L \ y)
    r(b) = Float64(maximum(abs.(big.(b) .- ref)) / maximum(abs.(ref)))
    κ = cond(L \ X)
    try
        β = gls(X, M, y)
        println("ε = $ε  cond(L⁻¹X) = $(round(κ, sigdigits=3))  rel.err ulm.lam = $(round(r(β), sigdigits=3))  rel.err QR = $(round(r(βqr), sigdigits=3))")
    catch e
        print("ε = $ε  cond(L⁻¹X) = $(round(κ, sigdigits=3))  ulm.lam: "); showerror(stdout, e)
        println("  rel.err QR = $(round(r(βqr), sigdigits=3))")
    end
end
