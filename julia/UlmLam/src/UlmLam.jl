"""
UlmLam: ulm.lam (Ulm Linear Algebra Mapping) for Julia.

Write linear algebra as in Julia; ulm.lam compiles it to C++ calling
ulmBLAS (or CBLAS/LAPACK) and calls the result:

    using UlmLam, LinearAlgebra

    @lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd),
                      y::Vec(:n))
        inv(X' * inv(M) * X) * X' * inv(M) * y
    end

    lib = UlmLam.compile(gls)
    b = lib.gls(X, M, y)

No inverse is formed: kernel selection maps the formula onto
factorisations and triangular solves (here: two Cholesky factorisations,
trsm, syrk, trsv).
"""
module UlmLam

using LinearAlgebra
using LinearAlgebra: UniformScaling
using Libdl
using SHA: sha256

export @lam, Mat, Vec, InOut, Index, Float, eye, assume, loop, extern,
       bind!, InfoError, LamError, Workspace, workspace_size, cpp, code_lam,
       @code_lam

include("frontend.jl")
include("runtime.jl")
include("codelam.jl")

end
