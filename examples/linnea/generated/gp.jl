using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Float64, ml3::Array{Float64,1}, ml4::Array{Float64,2})

Compute
mu = (Ks^T (K + (sigma_sq I(4000, 4000)))^-1 y)
Sigma = (Kss + (-1.0 Ks^T (K + (sigma_sq I(4000, 4000)))^-1 Ks)).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix Ks of size 4000 x 1000.
- `ml1::Array{Float64,2}`: Matrix K of size 4000 x 4000 with property SPSD.
- `ml2::Float64`: Scalar sigma_sq with property Positive.
- `ml3::Array{Float64,1}`: Vector y of size 4000.
- `ml4::Array{Float64,2}`: Matrix Kss of size 1000 x 1000 with property Symmetric.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Float64, ml3::Array{Float64,1}, ml4::Array{Float64,2})
    # cost: 6.13e+10 FLOPs
    # Ks: ml0, full, K: ml1, full, sigma_sq: ml2, full, y: ml3, full, Kss: ml4, full
    # tmp2 = (K + (sigma_sq I(4000, 4000)))
    d = ones(Float64, 4000);
    for i=1:length(d)
        ml1[i, i] += ml2*d[i];
    end;        

    # Ks: ml0, full, y: ml3, full, Kss: ml4, full, tmp2: ml1, full
    # (L3 L3^T) = tmp2
    LAPACK.potrf!('L', ml1)

    # Ks: ml0, full, y: ml3, full, Kss: ml4, full, L3: ml1, lower_triangular
    ml5 = Array{Float64}(undef, 4000, 1000)
    blascopy!(4000*1000, ml0, 1, ml5, 1)
    # tmp6 = (L3^-1 Ks)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # Ks: ml5, full, y: ml3, full, Kss: ml4, full, L3: ml1, lower_triangular, tmp6: ml0, full
    # tmp7 = (L3^-T tmp6)
    trsm!('L', 'L', 'T', 'N', 1.0, ml1, ml0)

    # Ks: ml5, full, y: ml3, full, Kss: ml4, full, tmp7: ml0, full
    ml6 = Array{Float64}(undef, 1000)
    # tmp8 = (tmp7^T y)
    gemv!('T', 1.0, ml0, ml3, 0.0, ml6)

    # Ks: ml5, full, Kss: ml4, full, tmp7: ml0, full, tmp8: ml6, full
    # tmp12 = (Kss + (-1.0 Ks^T tmp7))
    gemm!('T', 'N', -1.0, ml5, ml0, 1.0, ml4)

    # tmp8: ml6, full, tmp12: ml4, full
    # mu = tmp8
    # Sigma = tmp12
    return (ml6, ml4)
end