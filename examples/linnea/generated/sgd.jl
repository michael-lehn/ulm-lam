using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2}, ml3::Array{Float64,1}, ml4::Array{Float64,1}, ml5::Float64)

Compute
R = ((-1.0 Y) + (X W) + (ones b^T))
Wn = (W + (-1.0 step X^T R))
bn = ((-1.0 step R^T ones) + b).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix Y of size 1024 x 1024.
- `ml1::Array{Float64,2}`: Matrix X of size 1024 x 4096.
- `ml2::Array{Float64,2}`: Matrix W of size 4096 x 1024.
- `ml3::Array{Float64,1}`: Vector ones of size 1024.
- `ml4::Array{Float64,1}`: Vector b of size 1024.
- `ml5::Float64`: Scalar step with property Positive.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2}, ml3::Array{Float64,1}, ml4::Array{Float64,1}, ml5::Float64)
    # cost: 1.72e+10 FLOPs
    # Y: ml0, full, X: ml1, full, W: ml2, full, ones: ml3, full, b: ml4, full, step: ml5, full
    # tmp17 = ((-1.0 Y) + (X W))
    gemm!('N', 'N', 1.0, ml1, ml2, -1.0, ml0)

    # X: ml1, full, W: ml2, full, ones: ml3, full, b: ml4, full, step: ml5, full, tmp17: ml0, full
    # tmp6 = (tmp17 + (ones b^T))
    ger!(1.0, ml3, ml4, ml0)

    # X: ml1, full, W: ml2, full, ones: ml3, full, b: ml4, full, step: ml5, full, tmp6: ml0, full
    ml6 = Array{Float64}(undef, 1, 1)
    # tmp1 = (-1.0 step)
    ml6 = -1.0 * ml5

    # X: ml1, full, W: ml2, full, ones: ml3, full, b: ml4, full, tmp6: ml0, full, tmp1: ml6, full
    # tmp11 = (W + (tmp1 X^T tmp6))
    gemm!('T', 'N', ml6, ml1, ml0, 1.0, ml2)

    # ones: ml3, full, b: ml4, full, tmp6: ml0, full, tmp1: ml6, full, tmp11: ml2, full
    # tmp16 = ((tmp1 tmp6^T ones) + b)
    gemv!('T', ml6, ml0, ml3, 1.0, ml4)

    # tmp6: ml0, full, tmp11: ml2, full, tmp16: ml4, full
    # R = tmp6
    # Wn = tmp11
    # bn = tmp16
    return (ml0, ml2, ml4)
end