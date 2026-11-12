using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2})

Compute
O = (Q K^T V).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix Q of size 16384 x 64.
- `ml1::Array{Float64,2}`: Matrix K of size 16384 x 64.
- `ml2::Array{Float64,2}`: Matrix V of size 16384 x 65.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2})
    # cost: 2.73e+08 FLOPs
    # Q: ml0, full, K: ml1, full, V: ml2, full
    ml3 = Array{Float64}(undef, 64, 65)
    # tmp2 = (K^T V)
    gemm!('T', 'N', 1.0, ml1, ml2, 0.0, ml3)

    # Q: ml0, full, tmp2: ml3, full
    ml4 = Array{Float64}(undef, 16384, 65)
    # tmp3 = (Q tmp2)
    gemm!('N', 'N', 1.0, ml0, ml3, 0.0, ml4)

    # tmp3: ml4, full
    # O = tmp3
    return (ml4)
end