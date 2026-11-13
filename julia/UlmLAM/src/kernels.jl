# kernels.jl: the kernels that code generated for Julia calls.
#
# The Julia port of include/ulm_lam/cblas_backend.h: ulm.lam's lowering
# calls BLAS-like routines on strided views (incRow, incCol), and these map
# each call onto CBLAS (Basic Linear Algebra Subprograms, C interface) and
# LAPACK (Linear Algebra PACKage) through libblastrampoline, i.e. on the
# BLAS Julia has loaded: OpenBLAS by default, MKL (Intel Math Kernel
# Library) after `using MKL`, ulmBLAS after `UlmBLAS.forward()`.
#
# No copies of matrices: every operand is a row-major or column-major array
# (one stride is 1), and a row-major matrix is the transpose of a
# column-major one.  The adapter works in column-major order and turns the
# other orientation into a trans flag, a flipped triangle, or the
# transposed problem (C^T = B^T A^T).  Pointers are typed (Ptr{T}); offsets
# are in elements, as in C.

module Kernels

using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline

const ColMajor = Int32(102)
const NoTrans, Trans = Int32(111), Int32(112)
const Upper, Lower = Int32(121), Int32(122)
const NonUnit, Unit = Int32(131), Int32(132)
const Left, Right = Int32(141), Int32(142)

flip(t) = t == NoTrans ? Trans : NoTrans

# the element k of a typed pointer (C's p + k)
@inline at(p::Ptr{T}, k) where {T} = p + k * sizeof(T)

"""A rows x cols view with strides (incRow, incCol) as op(M) of a
column-major array M: col (op = NoTrans) or not (op = Trans), and M's
leading dimension."""
function orient(rows, cols, incRow, incCol)
    # (BLAS wants a leading dimension >= 1, also for empty matrices)
    if incRow == 1 && (cols <= 1 || incCol >= rows)
        return true, BlasInt(cols <= 1 ? max(rows, 1) : max(incCol, 1))
    end
    if incCol == 1
        return false, BlasInt(rows <= 1 ? max(cols, 1) : max(incRow, 1))
    end
    error("ulm.lam: a view that is neither row- nor column-major")
end
trans(col::Bool) = col ? NoTrans : Trans

"""Triangle of a symmetric view in the column-major array: a row-major view
stores its lower triangle in the array's upper one."""
triangle(lower, n, incRow, incCol) =
    lower == orient(n, n, incRow, incCol)[1] ? Lower : Upper

# ---- CBLAS and LAPACK, by element type --------------------------------------

for (T, p) in ((Float64, "d"), (Float32, "s"))
    sym(f) = Expr(:macrocall, Symbol("@blasfunc"), LineNumberNode(0), Symbol(f))
    gemm, gemv, symm, symv, syrk, trmm, trsm, trsv =
        (sym("cblas_$p$f") for f in ("gemm", "gemv", "symm", "symv", "syrk", "trmm", "trsm", "trsv"))
    potrf, potri, getrf, getrs, getri, gels, sytrf, sytrs =
        (sym("$p$(f)_") for f in ("potrf", "potri", "getrf", "getrs", "getri", "gels", "sytrf", "sytrs"))
    I = BlasInt
    @eval begin
        cgemm(o, ta, tb, m, n, k, al, A::Ptr{$T}, lda, B::Ptr{$T}, ldb, be, C::Ptr{$T}, ldc) =
            ccall(($gemm, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, $I, $I, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I, $T, Ptr{$T}, $I),
                  o, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc)
        cgemv(o, t, m, n, al, A::Ptr{$T}, lda, x::Ptr{$T}, ix, be, y::Ptr{$T}, iy) =
            ccall(($gemv, libblastrampoline), Cvoid,
                  (Int32, Int32, $I, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I, $T, Ptr{$T}, $I),
                  o, t, m, n, al, A, lda, x, ix, be, y, iy)
        csymm(o, s, u, m, n, al, A::Ptr{$T}, lda, B::Ptr{$T}, ldb, be, C::Ptr{$T}, ldc) =
            ccall(($symm, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, $I, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I, $T, Ptr{$T}, $I),
                  o, s, u, m, n, al, A, lda, B, ldb, be, C, ldc)
        csymv(o, u, n, al, A::Ptr{$T}, lda, x::Ptr{$T}, ix, be, y::Ptr{$T}, iy) =
            ccall(($symv, libblastrampoline), Cvoid,
                  (Int32, Int32, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I, $T, Ptr{$T}, $I),
                  o, u, n, al, A, lda, x, ix, be, y, iy)
        csyrk(o, u, t, n, k, al, A::Ptr{$T}, lda, be, C::Ptr{$T}, ldc) =
            ccall(($syrk, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, $I, $I, $T, Ptr{$T}, $I, $T, Ptr{$T}, $I),
                  o, u, t, n, k, al, A, lda, be, C, ldc)
        ctrmm(o, s, u, t, d, m, n, al, A::Ptr{$T}, lda, B::Ptr{$T}, ldb) =
            ccall(($trmm, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, Int32, Int32, $I, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I),
                  o, s, u, t, d, m, n, al, A, lda, B, ldb)
        ctrsm(o, s, u, t, d, m, n, al, A::Ptr{$T}, lda, B::Ptr{$T}, ldb) =
            ccall(($trsm, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, Int32, Int32, $I, $I, $T, Ptr{$T}, $I, Ptr{$T}, $I),
                  o, s, u, t, d, m, n, al, A, lda, B, ldb)
        ctrsv(o, u, t, d, n, A::Ptr{$T}, lda, x::Ptr{$T}, ix) =
            ccall(($trsv, libblastrampoline), Cvoid,
                  (Int32, Int32, Int32, Int32, $I, Ptr{$T}, $I, Ptr{$T}, $I),
                  o, u, t, d, n, A, lda, x, ix)
        function lpotrf(u::Char, n, A::Ptr{$T}, lda)
            info = Ref{$I}(0)
            ccall(($potrf, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  u, n, A, lda, info, 1)
            info[]
        end
        function lpotri_(u::Char, n, A::Ptr{$T}, lda)
            info = Ref{$I}(0)
            ccall(($potri, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  u, n, A, lda, info, 1)
            info[]
        end
        function lgetrf(m, n, A::Ptr{$T}, lda, piv::Ptr{$I})
            info = Ref{$I}(0)
            ccall(($getrf, libblastrampoline), Cvoid,
                  (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ref{$I}),
                  m, n, A, lda, piv, info)
            info[]
        end
        function lgetrs(t::Char, n, nrhs, A::Ptr{$T}, lda, piv::Ptr{$I}, B::Ptr{$T}, ldb)
            info = Ref{$I}(0)
            ccall(($getrs, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  t, n, nrhs, A, lda, piv, B, ldb, info, 1)
            nothing
        end
        # (LAPACK's workspace queried, then allocated here)
        function lgetri(n, A::Ptr{$T}, lda, piv::Ptr{$I})
            info, q = Ref{$I}(0), Ref{$T}(0)
            ccall(($getri, libblastrampoline), Cvoid,
                  (Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}),
                  n, A, lda, piv, q, -1, info)
            w = Vector{$T}(undef, max(Int(q[]), 1))
            ccall(($getri, libblastrampoline), Cvoid,
                  (Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}),
                  n, A, lda, piv, w, length(w), info)
            info[]
        end
        function lgels(t::Char, m, n, k, A::Ptr{$T}, lda, B::Ptr{$T}, ldb)
            info, q = Ref{$I}(0), Ref{$T}(0)
            ccall(($gels, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ref{$I},
                   Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  t, m, n, k, A, lda, B, ldb, q, -1, info, 1)
            w = Vector{$T}(undef, max(Int(q[]), 1))
            ccall(($gels, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ref{$I},
                   Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  t, m, n, k, A, lda, B, ldb, w, length(w), info, 1)
            info[]
        end
        function lsytrf(u::Char, n, A::Ptr{$T}, lda, piv::Ptr{$I})
            info, q = Ref{$I}(0), Ref{$T}(0)
            ccall(($sytrf, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  u, n, A, lda, piv, q, -1, info, 1)
            w = Vector{$T}(undef, max(Int(q[]), 1))
            ccall(($sytrf, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  u, n, A, lda, piv, w, length(w), info, 1)
            info[]
        end
        function lsytrs(u::Char, n, nrhs, A::Ptr{$T}, lda, piv::Ptr{$I}, B::Ptr{$T}, ldb)
            info = Ref{$I}(0)
            ccall(($sytrs, libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ptr{$T}, Ref{$I}, Ref{$I}, Clong),
                  u, n, nrhs, A, lda, piv, B, ldb, info, 1)
            nothing
        end
    end
end

# The inverse from a Cholesky factor.  An upper factor is transposed into a
# lower one first (n^2 copies): Accelerate's multithreaded dlauum with
# UPLO = 'U', which dpotri calls, gives wrong results nondeterministically
# (macOS 26.6); with UPLO = 'L' it is correct.
function lpotri(u::Char, n, A::Ptr{T}, lda) where {T}
    u == 'L' && return lpotri_(u, n, A, lda)
    l = Matrix{T}(undef, n, n)
    for j in 0:n-1, i in 0:j
        l[j+1, i+1] = unsafe_load(A, 1 + i + j * lda)
    end
    info = GC.@preserve l lpotri_('L', n, pointer(l), max(n, 1))
    for j in 0:n-1, i in 0:j
        unsafe_store!(A, l[j+1, i+1], 1 + i + j * lda)
    end
    info
end

"""B (rows x cols, strides incRow/incCol) copied into / from a column-major
array."""
function tocol(rows, cols, B::Ptr{T}, incRow, incCol) where {T}
    c = Matrix{T}(undef, rows, cols)
    for j in 0:cols-1, i in 0:rows-1
        c[i+1, j+1] = unsafe_load(B, 1 + i * incRow + j * incCol)
    end
    c
end
function fromcol!(B::Ptr{T}, c, incRow, incCol) where {T}
    rows, cols = size(c)
    for j in 0:cols-1, i in 0:rows-1
        unsafe_store!(B, c[i+1, j+1], 1 + i * incRow + j * incCol)
    end
end

# ---- the routines of the generated code -------------------------------------

# C := alpha A B + beta C (conj ignored: real types)
function gemm(m, n, k, alpha, _, A::Ptr{T}, incRowA, incColA, _, B::Ptr{T}, incRowB, incColB,
              beta, C::Ptr{T}, incRowC, incColC) where {T}
    acol, lda = orient(m, k, incRowA, incColA)
    bcol, ldb = orient(k, n, incRowB, incColB)
    ccol, ldc = orient(m, n, incRowC, incColC)
    if ccol
        cgemm(ColMajor, trans(acol), trans(bcol), m, n, k, T(alpha), A, lda, B, ldb, T(beta), C, ldc)
    else  # C^T = B^T A^T
        cgemm(ColMajor, flip(trans(bcol)), flip(trans(acol)), n, m, k, T(alpha), B, ldb, A, lda,
              T(beta), C, ldc)
    end
    nothing
end

# The lower (or upper) triangle of the n x n matrix C := beta C + alpha A B:
# gemm on blocks of columns, about half the flops of gemm.
function gemmtr(lower, n, k, alpha, _, A::Ptr{T}, incRowA, incColA, _, B::Ptr{T}, incRowB,
                incColB, beta, C::Ptr{T}, incRowC, incColC) where {T}
    nb = 128
    for j in 0:nb:n-1
        jb = min(nb, n - j)
        if lower  # C(j:n, j:j+jb)
            gemm(n - j, jb, k, alpha, false, at(A, j * incRowA), incRowA, incColA, false,
                 at(B, j * incColB), incRowB, incColB, beta, at(C, j * incRowC + j * incColC),
                 incRowC, incColC)
        else      # C(0:j+jb, j:j+jb)
            gemm(j + jb, jb, k, alpha, false, A, incRowA, incColA, false, at(B, j * incColB),
                 incRowB, incColB, beta, at(C, j * incColC), incRowC, incColC)
        end
    end
    nothing
end

# y := alpha A x + beta y
function gemv(m, n, alpha, A::Ptr{T}, incRowA, incColA, x::Ptr{T}, incX, beta, y::Ptr{T},
              incY) where {T}
    acol, lda = orient(m, n, incRowA, incColA)
    if acol
        cgemv(ColMajor, NoTrans, m, n, T(alpha), A, lda, x, incX, T(beta), y, incY)
    else
        cgemv(ColMajor, Trans, n, m, T(alpha), A, lda, x, incX, T(beta), y, incY)
    end
    nothing
end

# C := alpha A B + beta C, A symmetric (lower or upper triangle)
function symm(lower, m, n, alpha, A::Ptr{T}, incRowA, incColA, B::Ptr{T}, incRowB, incColB,
              beta, C::Ptr{T}, incRowC, incColC) where {T}
    _, lda = orient(m, m, incRowA, incColA)
    bcol, ldb = orient(m, n, incRowB, incColB)
    ccol, ldc = orient(m, n, incRowC, incColC)
    u = triangle(lower, m, incRowA, incColA)
    tmp = nothing
    if bcol != ccol  # B in the orientation of C
        tmp = ccol ? tocol(m, n, B, incRowB, incColB) : tocol(n, m, B, incColB, incRowB)
        ldb = BlasInt(ccol ? max(m, 1) : max(n, 1))
    end
    GC.@preserve tmp begin
        Bp = tmp === nothing ? B : pointer(tmp)
        if ccol
            csymm(ColMajor, Left, u, m, n, T(alpha), A, lda, Bp, ldb, T(beta), C, ldc)
        else  # C^T = B^T A
            csymm(ColMajor, Right, u, n, m, T(alpha), A, lda, Bp, ldb, T(beta), C, ldc)
        end
    end
    nothing
end
sylmm(m, n, alpha, A, iRA, iCA, B, iRB, iCB, beta, C, iRC, iCC) =
    symm(true, m, n, alpha, A, iRA, iCA, B, iRB, iCB, beta, C, iRC, iCC)
syumm(m, n, alpha, A, iRA, iCA, B, iRB, iCB, beta, C, iRC, iCC) =
    symm(false, m, n, alpha, A, iRA, iCA, B, iRB, iCB, beta, C, iRC, iCC)

# y := alpha A x + beta y, A symmetric, lower triangle
function sylmv(n, alpha, A::Ptr{T}, incRowA, incColA, x::Ptr{T}, incX, beta, y::Ptr{T},
               incY) where {T}
    _, lda = orient(n, n, incRowA, incColA)
    csymv(ColMajor, triangle(true, n, incRowA, incColA), n, T(alpha), A, lda, x, incX, T(beta),
          y, incY)
    nothing
end

# C := alpha A A^T + beta C, A n x k, the lower or upper triangle of C
function syrk(lower, n, k, alpha, A::Ptr{T}, incRowA, incColA, beta, C::Ptr{T}, incRowC,
              incColC) where {T}
    acol, lda = orient(n, k, incRowA, incColA)
    _, ldc = orient(n, n, incRowC, incColC)
    csyrk(ColMajor, triangle(lower, n, incRowC, incColC), trans(acol), n, k, T(alpha), A, lda,
          T(beta), C, ldc)
    nothing
end
sylrk(n, k, alpha, A, iRA, iCA, beta, C, iRC, iCC) =
    syrk(true, n, k, alpha, A, iRA, iCA, beta, C, iRC, iCC)
syurk(n, k, alpha, A, iRA, iCA, beta, C, iRC, iCC) =
    syrk(false, n, k, alpha, A, iRA, iCA, beta, C, iRC, iCC)

# B := alpha A^-1 B (trsm) or alpha A B (trmm), A m x m triangular, B m x n
for (name, c) in ((:trsm, :ctrsm), (:trmm, :ctrmm))
    @eval function $name(lower, m, n, alpha, unitDiag, A::Ptr{T}, incRowA, incColA, B::Ptr{T},
                         incRowB, incColB) where {T}
        acol, lda = orient(m, m, incRowA, incColA)
        bcol, ldb = orient(m, n, incRowB, incColB)
        u = lower == acol ? Lower : Upper  # triangle of the array
        d = unitDiag ? Unit : NonUnit
        if bcol
            $c(ColMajor, Left, u, trans(acol), d, m, n, T(alpha), A, lda, B, ldb)
        else  # X^T op(A)^T = B^T
            $c(ColMajor, Right, u, flip(trans(acol)), d, n, m, T(alpha), A, lda, B, ldb)
        end
        nothing
    end
end
trlmm(m, n, alpha, _, unit, A, iRA, iCA, B, iRB, iCB) = trmm(true, m, n, alpha, unit, A, iRA, iCA, B, iRB, iCB)
trumm(m, n, alpha, _, unit, A, iRA, iCA, B, iRB, iCB) = trmm(false, m, n, alpha, unit, A, iRA, iCA, B, iRB, iCB)
trlsm(m, n, alpha, _, unit, A, iRA, iCA, B, iRB, iCB) = trsm(true, m, n, alpha, unit, A, iRA, iCA, B, iRB, iCB)
trusm(m, n, alpha, _, unit, A, iRA, iCA, B, iRB, iCB) = trsm(false, m, n, alpha, unit, A, iRA, iCA, B, iRB, iCB)

# op(A) X = B (a zero on the diagonal is not checked: ulm.lam calls it
# after a successful factorisation)
function trtrs(lower, trans_, _, unitDiag, n, nrhs, A::Ptr{T}, incRowA, incColA, B::Ptr{T},
               incRowB, incColB) where {T}
    if trans_
        trsm(!lower, n, nrhs, one(T), unitDiag, A, incColA, incRowA, B, incRowB, incColB)
    else
        trsm(lower, n, nrhs, one(T), unitDiag, A, incRowA, incColA, B, incRowB, incColB)
    end
    0
end

# x := A^-1 x, A lower or upper triangular
function trsv(lower, n, unitDiag, A::Ptr{T}, incRowA, incColA, x::Ptr{T}, incX) where {T}
    acol, lda = orient(n, n, incRowA, incColA)
    u = lower == acol ? Lower : Upper
    ctrsv(ColMajor, u, trans(acol), unitDiag ? Unit : NonUnit, n, A, lda, x, incX)
    nothing
end
trlsv(n, unit, A, iRA, iCA, x, incX) = trsv(true, n, unit, A, iRA, iCA, x, incX)
trusv(n, unit, A, iRA, iCA, x, incX) = trsv(false, n, unit, A, iRA, iCA, x, incX)

# Y := alpha X + Y (no BLAS routine for strided matrices)
function geaxpy(m, n, alpha, X::Ptr{T}, incRowX, incColX, Y::Ptr{T}, incRowY, incColY) where {T}
    a = T(alpha)
    # the inner loop along the unit stride of Y
    if incColY == 1
        for i in 0:m-1, j in 0:n-1
            k = 1 + i * incRowY + j
            unsafe_store!(Y, unsafe_load(Y, k) + a * unsafe_load(X, 1 + i * incRowX + j * incColX), k)
        end
    else
        for j in 0:n-1, i in 0:m-1
            k = 1 + i * incRowY + j * incColY
            unsafe_store!(Y, unsafe_load(Y, k) + a * unsafe_load(X, 1 + i * incRowX + j * incColX), k)
        end
    end
    nothing
end

# X := alpha X; as BLAS's scal, alpha = 0 sets X to zero (also NaN and Inf)
function gescal(m, n, alpha, X::Ptr{T}, incRowX, incColX) where {T}
    a = T(alpha)
    for i in 0:m-1, j in 0:n-1
        k = 1 + i * incRowX + j * incColX
        unsafe_store!(X, a == 0 ? zero(T) : unsafe_load(X, k) * a, k)
    end
    nothing
end

# A = L L^T (lower) or U^T U, in place; returns LAPACK's info
function potrf(lower, n, A::Ptr{T}, incRowA, incColA) where {T}
    _, lda = orient(n, n, incRowA, incColA)
    Int64(lpotrf(triangle(lower, n, incRowA, incColA) == Lower ? 'L' : 'U', n, A, lda))
end

# the inverse from the Cholesky factor of potrf (same uplo and strides)
function potri(lower, n, A::Ptr{T}, incRowA, incColA) where {T}
    _, lda = orient(n, n, incRowA, incColA)
    Int64(lpotri(triangle(lower, n, incRowA, incColA) == Lower ? 'L' : 'U', n, A, lda))
end

# P A = L U in place; a row-major A is factorised as the column-major A^T.
# The pivots are LAPACK's (1-based), stored in the IR's integer type.
function getrf(m, n, A::Ptr{T}, incRowA, incColA, piv::Ptr{P}, _) where {T,P}
    acol, lda = orient(m, n, incRowA, incColA)
    p = Vector{BlasInt}(undef, min(m, n))
    info = GC.@preserve p acol ? lgetrf(m, n, A, lda, pointer(p)) : lgetrf(n, m, A, lda, pointer(p))
    for k in eachindex(p)
        unsafe_store!(piv, P(p[k]), k)
    end
    Int64(info)
end

pivots(piv::Ptr{P}, n) where {P} = BlasInt[unsafe_load(piv, k) for k in 1:n]

# B := op(A)^-1 B with the factors of getrf
function getrs(trans_, _, n, nrhs, LU::Ptr{T}, incRowA, incColA, piv::Ptr, _, B::Ptr{T},
               incRowB, incColB) where {T}
    acol, lda = orient(n, n, incRowA, incColA)
    bcol, ldb = orient(n, nrhs, incRowB, incColB)
    t = trans_ == acol ? 'T' : 'N'  # factors of A (column-major) or of A^T
    p = pivots(piv, n)
    GC.@preserve p begin
        if bcol
            lgetrs(t, n, nrhs, LU, lda, pointer(p), B, ldb)
        else
            c = tocol(n, nrhs, B, incRowB, incColB)
            GC.@preserve c lgetrs(t, n, nrhs, LU, lda, pointer(p), pointer(c), max(n, 1))
            fromcol!(B, c, incRowB, incColB)
        end
    end
    nothing
end

# A^-1 from the factors of getrf (same strides), in place
function getri(n, A::Ptr{T}, incRowA, incColA, piv::Ptr, _) where {T}
    _, lda = orient(n, n, incRowA, incColA)
    p = pivots(piv, n)
    Int64(GC.@preserve p lgetri(n, A, lda, pointer(p)))
end

# Least squares by QR: B (m x k) := the minimiser of ||A X - B|| in its
# leading n rows; a row-major A is the column-major A^T (gels with 'T').
function gels(m, n, k, A::Ptr{T}, incRowA, incColA, B::Ptr{T}, incRowB, incColB) where {T}
    acol, lda = orient(m, n, incRowA, incColA)
    c = tocol(m, k, B, incRowB, incColB)
    info = GC.@preserve c lgels(acol ? 'N' : 'T', acol ? m : n, acol ? n : m, k, A, lda,
                                pointer(c), max(m, 1))
    fromcol!(B, c, incRowB, incColB)
    Int64(info)
end

# A = L D L^T (Bunch-Kaufman), in place
function sytrf(_, n, A::Ptr{T}, incRowA, incColA, piv::Ptr{P}, _) where {T,P}
    acol, lda = orient(n, n, incRowA, incColA)
    p = Vector{BlasInt}(undef, n)
    info = GC.@preserve p lsytrf(acol ? 'L' : 'U', n, A, lda, pointer(p))
    for k in eachindex(p)
        unsafe_store!(piv, P(p[k]), k)
    end
    Int64(info)
end

# B := A^-1 B with the factors of sytrf
function sytrs(_, _, n, nrhs, LDL::Ptr{T}, incRowA, incColA, piv::Ptr, _, B::Ptr{T}, incRowB,
               incColB) where {T}
    acol, lda = orient(n, n, incRowA, incColA)
    bcol, ldb = orient(n, nrhs, incRowB, incColB)
    u = acol ? 'L' : 'U'
    p = pivots(piv, n)
    GC.@preserve p begin
        if bcol
            lsytrs(u, n, nrhs, LDL, lda, pointer(p), B, ldb)
        else
            c = tocol(n, nrhs, B, incRowB, incColB)
            GC.@preserve c lsytrs(u, n, nrhs, LDL, lda, pointer(p), pointer(c), max(n, 1))
            fromcol!(B, c, incRowB, incColB)
        end
    end
    nothing
end

# ---- the helpers the lowering emits as C++ (ulm_lam_*) ----------------------

# memref.copy
copy(dst::Ptr, src::Ptr, bytes) =
    (ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, bytes); nothing)

# B := op(A) (ulm_lamk.gecopy), blocked
function gecopy(m, n, A::Ptr{T}, incRowA, incColA, B::Ptr{T}, incRowB, incColB) where {T}
    bs = 64
    for i0 in 0:bs:m-1, j0 in 0:bs:n-1
        for i in i0:min(i0 + bs, m)-1, j in j0:min(j0 + bs, n)-1
            unsafe_store!(B, unsafe_load(A, 1 + i * incRowA + j * incColA), 1 + i * incRowB + j * incColB)
        end
    end
    nothing
end

# A := A + alpha I
function adddiag(n, alpha, A::Ptr{T}, incRow, incCol) where {T}
    for i in 0:n-1
        k = 1 + i * (incRow + incCol)
        unsafe_store!(A, unsafe_load(A, k) + T(alpha), k)
    end
    nothing
end

# mirror the stored triangle of a symmetric matrix, in tiles
function sycomplete(lower, n, A::Ptr{T}, incRow, incCol) where {T}
    b = 64
    for ib in 0:b:n-1, jb in 0:b:ib
        for i in ib:min(n, ib + b)-1, j in jb:min(i, jb + b)-1
            lo, up = 1 + i * incRow + j * incCol, 1 + j * incRow + i * incCol
            lower ? unsafe_store!(A, unsafe_load(A, lo), up) : unsafe_store!(A, unsafe_load(A, up), lo)
        end
    end
    nothing
end

# aligned_alloc and free of the generated code (temporaries the caller did
# not provide a workspace for)
aligned_alloc(align, bytes) = Libc.malloc(max(bytes, 1))
free(p) = Libc.free(p)

end # module Kernels
