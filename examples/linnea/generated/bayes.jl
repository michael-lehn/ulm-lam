using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Float64, ml1::Float64, ml2::Array{Float64,2}, ml3::Array{Float64,1}, ml4::Array{Float64,2}, ml5::Float64)

Compute
w = (beta ((alpha I(2000, 2000)) + (beta Phi^T Phi))^-1 Phi^T t)
mean = (Phis w)
cov = ((betainv I(1379, 1379)) + (Phis ((alpha I(2000, 2000)) + (beta Phi^T Phi))^-1 Phis^T)).

Requires at least Julia v1.0.

# Arguments
- `ml0::Float64`: Scalar beta with property Positive.
- `ml1::Float64`: Scalar alpha with property Positive.
- `ml2::Array{Float64,2}`: Matrix Phi of size 16000 x 2000 with property FullRank.
- `ml3::Array{Float64,1}`: Vector t of size 16000.
- `ml4::Array{Float64,2}`: Matrix Phis of size 1379 x 2000 with property FullRank.
- `ml5::Float64`: Scalar betainv with property Positive.
"""                    
function algorithm0(ml0::Float64, ml1::Float64, ml2::Array{Float64,2}, ml3::Array{Float64,1}, ml4::Array{Float64,2}, ml5::Float64)
    # cost: 7.61e+10 FLOPs
    # beta: ml0, full, alpha: ml1, full, Phi: ml2, full, t: ml3, full, Phis: ml4, full, betainv: ml5, full
    ml6 = Array{Float64}(I, 2000, 2000)
    # tmp1 = ((alpha I(2000, 2000)) + (beta Phi^T Phi))
    syrk!('L', 'T', ml0, ml2, ml1, ml6)

    # beta: ml0, full, Phi: ml2, full, t: ml3, full, Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, tmp1: ml6, symmetric_lower_triangular
    # (L7 L7^T) = tmp1
    LAPACK.potrf!('L', ml6)

    # beta: ml0, full, Phi: ml2, full, t: ml3, full, Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular
    ml7 = Array{Float64}(undef, 2000)
    # tmp14 = (Phi^T t)
    gemv!('T', 1.0, ml2, ml3, 0.0, ml7)

    # beta: ml0, full, Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular, tmp14: ml7, full
    # tmp17 = (L7^-1 tmp14)
    trsv!('L', 'N', 'N', ml6, ml7)

    # beta: ml0, full, Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular, tmp17: ml7, full
    # tmp20 = (L7^-T tmp17)
    trsv!('L', 'T', 'N', ml6, ml7)

    # beta: ml0, full, Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular, tmp20: ml7, full
    # tmp21 = (beta tmp20)
    scal!(2000, ml0, ml7, 1)

    # Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular, tmp21: ml7, full
    ml8 = Array{Float64}(undef, 1379)
    # tmp22 = (Phis tmp21)
    gemv!('N', 1.0, ml4, ml7, 0.0, ml8)

    # Phis: ml4, full, betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, L7: ml6, lower_triangular, tmp21: ml7, full, tmp22: ml8, full
    # tmp24 = (Phis L7^-T)
    trsm!('R', 'L', 'T', 'N', 1.0, ml6, ml4)

    # betainv: ml5, full, I(2000, 2000): ml6, symmetric_lower_triangular, tmp21: ml7, full, tmp22: ml8, full, tmp24: ml4, full
    ml9 = Array{Float64}(I, 1379, 1379)
    # tmp27 = ((betainv I(1379, 1379)) + (tmp24 tmp24^T))
    syrk!('L', 'N', 1.0, ml4, ml5, ml9)

    # I(2000, 2000): ml6, symmetric_lower_triangular, tmp21: ml7, full, tmp22: ml8, full, I(1379, 1379): ml9, symmetric_lower_triangular, tmp27: ml9, symmetric_lower_triangular
    for i = 1:1379-1;
        view(ml9, i, i+1:1379)[:] = view(ml9, i+1:1379, i);
    end;

    # I(2000, 2000): ml6, symmetric_lower_triangular, tmp21: ml7, full, tmp22: ml8, full, I(1379, 1379): ml9, symmetric_lower_triangular, tmp27: ml9, full
    # w = tmp21
    # mean = tmp22
    # cov = tmp27
    return (ml7, ml8, ml9)
end