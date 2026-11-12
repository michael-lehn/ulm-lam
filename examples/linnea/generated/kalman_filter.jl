using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,1}, ml2::Array{Float64,2}, ml3::Array{Float64,2}, ml4::Array{Float64,2}, ml5::Array{Float64,2}, ml6::Array{Float64,1})

Compute
xp = (F x)
Pp = (Q + (F P F^T))
Kg = (Pp H^T (R + (H Pp H^T))^-1)
xn = ((Kg ((-1.0 H xp) + z)) + xp)
Pn = (Pp + (-1.0 Kg H Pp)).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix F of size 1000 x 1000 with property Non-singular.
- `ml1::Array{Float64,1}`: Vector x of size 1000.
- `ml2::Array{Float64,2}`: Matrix Q of size 1000 x 1000 with property SPSD.
- `ml3::Array{Float64,2}`: Matrix P of size 1000 x 1000 with property SPD.
- `ml4::Array{Float64,2}`: Matrix H of size 100 x 1000 with property FullRank.
- `ml5::Array{Float64,2}`: Matrix R of size 100 x 100 with property SPD.
- `ml6::Array{Float64,1}`: Vector z of size 100.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,1}, ml2::Array{Float64,2}, ml3::Array{Float64,2}, ml4::Array{Float64,2}, ml5::Array{Float64,2}, ml6::Array{Float64,1})
    # cost: 2.78e+09 FLOPs
    # F: ml0, full, x: ml1, full, Q: ml2, full, P: ml3, full, H: ml4, full, R: ml5, full, z: ml6, full
    ml7 = Array{Float64}(undef, 1000)
    # tmp2 = (F x)
    gemv!('N', 1.0, ml0, ml1, 0.0, ml7)

    # F: ml0, full, Q: ml2, full, P: ml3, full, H: ml4, full, R: ml5, full, z: ml6, full, tmp2: ml7, full
    # (L70 L70^T) = P
    LAPACK.potrf!('L', ml3)

    # F: ml0, full, Q: ml2, full, H: ml4, full, R: ml5, full, z: ml6, full, tmp2: ml7, full, L70: ml3, lower_triangular
    # tmp71 = (F L70)
    trmm!('R', 'L', 'N', 'N', 1.0, ml3, ml0)

    # Q: ml2, full, H: ml4, full, R: ml5, full, z: ml6, full, tmp2: ml7, full, tmp71: ml0, full
    # tmp5 = (Q + (tmp71 tmp71^T))
    syrk!('L', 'N', 1.0, ml0, 1.0, ml2)

    # H: ml4, full, R: ml5, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular
    ml8 = Array{Float64}(undef, 100, 1000)
    # tmp6 = (H tmp5)
    symm!('R', 'L', 1.0, ml2, ml4, 0.0, ml8)

    # H: ml4, full, R: ml5, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full
    # tmp8 = (R + (tmp6 H^T))
    gemm!('N', 'T', 1.0, ml8, ml4, 1.0, ml5)

    # H: ml4, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp8: ml5, full
    ml9 = Array{Float64}(undef, 1000, 100)
    # tmp31 = tmp6^T
    transpose!(ml9, ml8)

    # H: ml4, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp8: ml5, full, tmp31: ml9, full
    # (L9 L9^T) = tmp8
    LAPACK.potrf!('L', ml5)

    # H: ml4, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp31: ml9, full, L9: ml5, lower_triangular
    # tmp32 = (tmp31 L9^-T)
    trsm!('R', 'L', 'T', 'N', 1.0, ml5, ml9)

    # H: ml4, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, L9: ml5, lower_triangular, tmp32: ml9, full
    # tmp16 = (tmp32 L9^-1)
    trsm!('R', 'L', 'N', 'N', 1.0, ml5, ml9)

    # H: ml4, full, z: ml6, full, tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp16: ml9, full
    # tmp20 = ((-1.0 H tmp2) + z)
    gemv!('N', -1.0, ml4, ml7, 1.0, ml6)

    # tmp2: ml7, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp16: ml9, full, tmp20: ml6, full
    ml10 = Array{Float64}(undef, 1000)
    blascopy!(1000, ml7, 1, ml10, 1)
    # tmp22 = ((tmp16 tmp20) + tmp2)
    gemv!('N', 1.0, ml9, ml6, 1.0, ml7)

    # tmp2: ml10, full, tmp5: ml2, symmetric_lower_triangular, tmp6: ml8, full, tmp16: ml9, full, tmp22: ml7, full
    for i = 1:1000-1;
        view(ml2, i, i+1:1000)[:] = view(ml2, i+1:1000, i);
    end;
    ml11 = Array{Float64}(undef, 1000, 1000)
    blascopy!(1000*1000, ml2, 1, ml11, 1)
    # tmp27 = (tmp5 + (-1.0 tmp16 tmp6))
    gemm!('N', 'N', -1.0, ml9, ml8, 1.0, ml2)

    # tmp2: ml10, full, tmp5: ml11, full, tmp16: ml9, full, tmp22: ml7, full, tmp27: ml2, full
    # xp = tmp2
    # Pp = tmp5
    # Kg = tmp16
    # xn = tmp22
    # Pn = tmp27
    return (ml10, ml11, ml9, ml7, ml2)
end