"""
UlmLAM: ulm.lam (Ulm Linear Algebra Mapping) for Julia.

Write linear algebra as in Julia; ulm.lam compiles it to calls of BLAS and
LAPACK (Julia code on Julia's own BLAS by default, or C++ on ulmBLAS) and
calls the result:

    using UlmLAM, LinearAlgebra

    @lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd),
                      y::Vec(:n))
        inv(X' * inv(M) * X) * X' * inv(M) * y
    end

    lib = UlmLAM.compile(gls)
    b = lib.gls(X, M, y)

No inverse is formed: kernel selection maps the formula onto
factorisations and triangular solves (here: two Cholesky factorisations,
trsm, syrk, trsv).
"""
module UlmLAM

using LinearAlgebra
using LinearAlgebra: UniformScaling
using Libdl
using SHA: sha256
import Scratch

export @lam, update!, Mat, Vec, InOut, Index, Float, eye, assume, loop, extern,
       bind!, InfoError, LamError, Workspace, workspace_size, cpp, code_lam, lstsq,
       @code_lam

include("kernels.jl")
include("frontend.jl")
include("runtime.jl")
include("codelam.jl")

end
