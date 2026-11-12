using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
b = ((X^T M^-1 X)^-1 X^T M^-1 y).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X of size 2000 x 200 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix M of size 2000 x 2000 with property SPD.
- `ml2::Array{Float64,1}`: Vector y of size 2000.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    # cost: 3.55e+09 FLOPs
    # X: ml0, full, M: ml1, full, y: ml2, full
    # (L2 L2^T) = M
    LAPACK.potrf!('L', ml1)

    # X: ml0, full, y: ml2, full, L2: ml1, lower_triangular
    # tmp38 = (L2^-1 y)
    trsv!('L', 'N', 'N', ml1, ml2)

    # X: ml0, full, L2: ml1, lower_triangular, tmp38: ml2, full
    # tmp12 = (L2^-1 X)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # tmp38: ml2, full, tmp12: ml0, full
    ml3 = Array{Float64}(undef, 200, 200)
    # tmp14 = (tmp12^T tmp12)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml3)

    # tmp38: ml2, full, tmp12: ml0, full, tmp14: ml3, symmetric_lower_triangular
    # (L15 L15^T) = tmp14
    LAPACK.potrf!('L', ml3)

    # tmp38: ml2, full, tmp12: ml0, full, L15: ml3, lower_triangular
    ml4 = Array{Float64}(undef, 200)
    # tmp21 = (tmp12^T tmp38)
    gemv!('T', 1.0, ml0, ml2, 0.0, ml4)

    # L15: ml3, lower_triangular, tmp21: ml4, full
    # tmp23 = (L15^-1 tmp21)
    trsv!('L', 'N', 'N', ml3, ml4)

    # L15: ml3, lower_triangular, tmp23: ml4, full
    # tmp24 = (L15^-T tmp23)
    trsv!('L', 'T', 'N', ml3, ml4)

    # tmp24: ml4, full
    # b = tmp24
    return (ml4)
end