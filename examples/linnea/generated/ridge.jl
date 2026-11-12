using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Float64, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
beta = (((lambda I(2000, 2000)) + (X^T X))^-1 X^T y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Float64`: Scalar lambda with property Positive.
- `ml1::Array{Float64,2}`: Matrix X of size 16000 x 2000 with property FullRank.
- `ml2::Array{Float64,1}`: Vector y of size 16000.
"""                    
function algorithm0(ml0::Float64, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 6.67e+10 FLOPs
    # lambda: ml0, full, X: ml1, full, y: ml2, full
    ml3 = Array{Float64}(undef, 2000)
    # tmp10 = (X^T y)
    gemv!('T', 1.0, ml1, ml2, 0.0, ml3)

    # lambda: ml0, full, X: ml1, full, tmp10: ml3, full
    ml4 = Array{Float64}(I, 2000, 2000)
    # tmp3 = ((lambda I(2000, 2000)) + (X^T X))
    syrk!('L', 'T', 1.0, ml1, ml0, ml4)

    # tmp10: ml3, full, I(2000, 2000): ml4, symmetric_lower_triangular, tmp3: ml4, symmetric_lower_triangular
    # (L4 L4^T) = tmp3
    LAPACK.potrf!('L', ml4)

    # tmp10: ml3, full, I(2000, 2000): ml4, symmetric_lower_triangular, L4: ml4, lower_triangular
    # tmp12 = (L4^-1 tmp10)
    trsv!('L', 'N', 'N', ml4, ml3)

    # I(2000, 2000): ml4, symmetric_lower_triangular, L4: ml4, lower_triangular, tmp12: ml3, full
    # tmp13 = (L4^-T tmp12)
    trsv!('L', 'T', 'N', ml4, ml3)

    # I(2000, 2000): ml4, symmetric_lower_triangular, tmp13: ml3, full
    # beta = tmp13
    return (ml3)
end