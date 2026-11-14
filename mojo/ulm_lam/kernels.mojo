# kernels.mojo: the kernels that code generated for Mojo calls.
#
# The Mojo port of julia/UlmLAM/src/kernels.jl (itself a port of
# include/ulm_lam/cblas_backend.h): ulm.lam's lowering calls BLAS-like
# routines on strided views (incRow, incCol), and these map each call onto
# CBLAS (C interface to the BLAS (Basic Linear Algebra Subprograms)) and
# LAPACK (Linear Algebra PACKage) of Apple Accelerate.  LAPACK is called
# under its $NEWLAPACK symbols (LAPACK 3.9-ish, 32-bit integers); the plain
# names (dpotrf_) are Accelerate's old CLAPACK.
#
# No copies of matrices: every operand is a row-major or column-major array
# (one stride is 1), and a row-major matrix is the transpose of a
# column-major one.  The adapter works in column-major order and turns the
# other orientation into a trans flag, a flipped triangle, or the
# transposed problem (C^T = B^T A^T).  Exceptions, with a temporary
# column-major copy: symm with B and C in different orientations, getrs,
# sytrs and gels with a row-major right-hand side, potri of an upper
# factor, and the workspaces of LAPACK's getri, gels and sytrf.
#
# Pointers are typed; offsets are in elements, as in C.  The element type T
# (float32 or float64) is inferred from the pointers; scalars of the
# generated code (alpha, beta) are converted to it.

from std.ffi import external_call
from std.os import abort
from std.sys import size_of

comptime P[T: DType] = MutPointer[Scalar[T], MutUntrackedOrigin]
"""A raw pointer to elements of type T (the pointers of generated code)."""

comptime ColMajor = Int32(102)
comptime NoTrans = Int32(111)
comptime Trans = Int32(112)
comptime Upper = Int32(121)
comptime Lower = Int32(122)
comptime NonUnit = Int32(131)
comptime Unit = Int32(132)
comptime Left = Int32(141)
comptime Right = Int32(142)


def flip(t: Int32) -> Int32:
    return Trans if t == NoTrans else NoTrans


def trans(col: Bool) -> Int32:
    return NoTrans if col else Trans


def orient(
    rows: Int, cols: Int, incRow: Int, incCol: Int
) -> Tuple[Bool, Int32]:
    """A rows x cols view with strides (incRow, incCol) as op(M) of a
    column-major array M: column-major (op = NoTrans) or not (op = Trans),
    and M's leading dimension."""
    # (BLAS wants a leading dimension >= 1, also for empty matrices)
    if incRow == 1 and (cols <= 1 or incCol >= rows):
        return (True, Int32(max(rows, 1) if cols <= 1 else max(incCol, 1)))
    if incCol != 1:
        abort("ulm.lam: a view that is neither row- nor column-major")
    return (False, Int32(max(cols, 1) if rows <= 1 else max(incRow, 1)))


def triangle(lower: Bool, n: Int, incRow: Int, incCol: Int) -> Int32:
    """The triangle of a symmetric view in the column-major array: a
    row-major view stores its lower triangle in the array's upper one."""
    return Lower if lower == orient(n, n, incRow, incCol)[0] else Upper


def char(c: StringLiteral) -> Int8:
    return Int8(ord(c))


def uplo_char(u: Int32) -> Int8:
    return char("L") if u == Lower else char("U")


# ---- memory ------------------------------------------------------------------


def aligned_alloc(align: Int, bytes: Int) -> P[DType.int8]:
    """Temporaries the caller did not provide a workspace for."""
    var size = (max(bytes, 1) + align - 1) // align * align
    var p = external_call["aligned_alloc", Int](align, size)
    if p == 0:
        abort("ulm.lam: out of memory")
    return P[DType.int8](unsafe_from_address=p)


def free(p: P[DType.int8]):
    external_call["free", NoneType](p)


def buffer[T: DType](count: Int) -> P[T]:
    """A temporary of count elements (released with release)."""
    return aligned_alloc(64, count * size_of[T]()).unsafe_bitcast[Scalar[T]]()


def release[T: DType](p: P[T]):
    free(p.unsafe_bitcast[Int8]())


def tocol[
    T: DType
](rows: Int, cols: Int, B: P[T], incRow: Int, incCol: Int) -> P[T]:
    """B (rows x cols, strides incRow/incCol) copied into a new
    column-major array."""
    var c = buffer[T](rows * cols)
    for j in range(cols):
        for i in range(rows):
            c[unsafe_offset=i + j * rows] = B[
                unsafe_offset=i * incRow + j * incCol
            ]
    return c


def fromcol[
    T: DType
](B: P[T], c: P[T], rows: Int, cols: Int, incRow: Int, incCol: Int):
    """The column-major array c copied back into B, then released."""
    for j in range(cols):
        for i in range(rows):
            B[unsafe_offset=i * incRow + j * incCol] = c[
                unsafe_offset=i + j * rows
            ]
    release(c)


# ---- CBLAS -------------------------------------------------------------------
#
# (the symbol chosen at compile time from the element type: s for float32,
# d for float64; the legacy cblas_* symbols take 32-bit integers)


def cgemm[
    T: DType
](
    ta: Int32,
    tb: Int32,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    B: P[T],
    ldb: Int32,
    beta: Scalar[T],
    C: P[T],
    ldc: Int32,
):
    comptime name = "cblas_dgemm" if T == DType.float64 else "cblas_sgemm"
    external_call[name, NoneType](
        ColMajor,
        ta,
        tb,
        Int32(m),
        Int32(n),
        Int32(k),
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    )


def cgemv[
    T: DType
](
    t: Int32,
    m: Int,
    n: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    x: P[T],
    incX: Int,
    beta: Scalar[T],
    y: P[T],
    incY: Int,
):
    comptime name = "cblas_dgemv" if T == DType.float64 else "cblas_sgemv"
    external_call[name, NoneType](
        ColMajor,
        t,
        Int32(m),
        Int32(n),
        alpha,
        A,
        lda,
        x,
        Int32(incX),
        beta,
        y,
        Int32(incY),
    )


def csymm[
    T: DType
](
    s: Int32,
    u: Int32,
    m: Int,
    n: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    B: P[T],
    ldb: Int32,
    beta: Scalar[T],
    C: P[T],
    ldc: Int32,
):
    comptime name = "cblas_dsymm" if T == DType.float64 else "cblas_ssymm"
    external_call[name, NoneType](
        ColMajor, s, u, Int32(m), Int32(n), alpha, A, lda, B, ldb, beta, C, ldc
    )


def csymv[
    T: DType
](
    u: Int32,
    n: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    x: P[T],
    incX: Int,
    beta: Scalar[T],
    y: P[T],
    incY: Int,
):
    comptime name = "cblas_dsymv" if T == DType.float64 else "cblas_ssymv"
    external_call[name, NoneType](
        ColMajor,
        u,
        Int32(n),
        alpha,
        A,
        lda,
        x,
        Int32(incX),
        beta,
        y,
        Int32(incY),
    )


def csyrk[
    T: DType
](
    u: Int32,
    t: Int32,
    n: Int,
    k: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    beta: Scalar[T],
    C: P[T],
    ldc: Int32,
):
    comptime name = "cblas_dsyrk" if T == DType.float64 else "cblas_ssyrk"
    external_call[name, NoneType](
        ColMajor, u, t, Int32(n), Int32(k), alpha, A, lda, beta, C, ldc
    )


def ctrmm[
    T: DType
](
    s: Int32,
    u: Int32,
    t: Int32,
    d: Int32,
    m: Int,
    n: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    B: P[T],
    ldb: Int32,
):
    comptime name = "cblas_dtrmm" if T == DType.float64 else "cblas_strmm"
    external_call[name, NoneType](
        ColMajor, s, u, t, d, Int32(m), Int32(n), alpha, A, lda, B, ldb
    )


def ctrsm[
    T: DType
](
    s: Int32,
    u: Int32,
    t: Int32,
    d: Int32,
    m: Int,
    n: Int,
    alpha: Scalar[T],
    A: P[T],
    lda: Int32,
    B: P[T],
    ldb: Int32,
):
    comptime name = "cblas_dtrsm" if T == DType.float64 else "cblas_strsm"
    external_call[name, NoneType](
        ColMajor, s, u, t, d, Int32(m), Int32(n), alpha, A, lda, B, ldb
    )


def ctrsv[
    T: DType
](
    u: Int32,
    t: Int32,
    d: Int32,
    n: Int,
    A: P[T],
    lda: Int32,
    x: P[T],
    incX: Int,
):
    comptime name = "cblas_dtrsv" if T == DType.float64 else "cblas_strsv"
    external_call[name, NoneType](
        ColMajor, u, t, d, Int32(n), A, lda, x, Int32(incX)
    )


# ---- LAPACK (Fortran convention: every argument by pointer) ------------------
#
# Each argument is its own local variable: Mojo does not pass one variable
# twice by pointer.  Accelerate needs no hidden string lengths.


def lpotrf[T: DType](u: Int32, n: Int, A: P[T], lda: Int32) -> Int:
    comptime name = "dpotrf$NEWLAPACK" if T == DType.float64 else "spotrf$NEWLAPACK"
    var uplo = uplo_char(u)
    var n_ = Int32(n)
    var ld = lda
    var info = Int32(0)
    external_call[name, NoneType](
        Pointer(to=uplo), Pointer(to=n_), A, Pointer(to=ld), Pointer(to=info)
    )
    return Int(info)


def lpotri_[T: DType](u: Int32, n: Int, A: P[T], lda: Int32) -> Int:
    comptime name = "dpotri$NEWLAPACK" if T == DType.float64 else "spotri$NEWLAPACK"
    var uplo = uplo_char(u)
    var n_ = Int32(n)
    var ld = lda
    var info = Int32(0)
    external_call[name, NoneType](
        Pointer(to=uplo), Pointer(to=n_), A, Pointer(to=ld), Pointer(to=info)
    )
    return Int(info)


def lgetrf[
    T: DType
](m: Int, n: Int, A: P[T], lda: Int32, piv: P[DType.int32]) -> Int:
    comptime name = "dgetrf$NEWLAPACK" if T == DType.float64 else "sgetrf$NEWLAPACK"
    var m_ = Int32(m)
    var n_ = Int32(n)
    var ld = lda
    var info = Int32(0)
    external_call[name, NoneType](
        Pointer(to=m_), Pointer(to=n_), A, Pointer(to=ld), piv, Pointer(to=info)
    )
    return Int(info)


def lgetrs[
    T: DType
](
    t: Int8,
    n: Int,
    nrhs: Int,
    A: P[T],
    lda: Int32,
    piv: P[DType.int32],
    B: P[T],
    ldb: Int32,
) -> Int:
    comptime name = "dgetrs$NEWLAPACK" if T == DType.float64 else "sgetrs$NEWLAPACK"
    var t_ = t
    var n_ = Int32(n)
    var k_ = Int32(nrhs)
    var la = lda
    var lb = ldb
    var info = Int32(0)
    external_call[name, NoneType](
        Pointer(to=t_),
        Pointer(to=n_),
        Pointer(to=k_),
        A,
        Pointer(to=la),
        piv,
        B,
        Pointer(to=lb),
        Pointer(to=info),
    )
    return Int(info)


def lgetri[T: DType](n: Int, A: P[T], lda: Int32, piv: P[DType.int32]) -> Int:
    """(LAPACK's workspace queried, then allocated here.)"""
    comptime name = "dgetri$NEWLAPACK" if T == DType.float64 else "sgetri$NEWLAPACK"
    var n_ = Int32(n)
    var ld = lda
    var info = Int32(0)
    var q = Scalar[T](0)
    var lw = Int32(-1)
    external_call[name, NoneType](
        Pointer(to=n_),
        A,
        Pointer(to=ld),
        piv,
        Pointer(to=q),
        Pointer(to=lw),
        Pointer(to=info),
    )
    lw = Int32(max(Int(q), 1))
    var w = buffer[T](Int(lw))
    external_call[name, NoneType](
        Pointer(to=n_),
        A,
        Pointer(to=ld),
        piv,
        w,
        Pointer(to=lw),
        Pointer(to=info),
    )
    release(w)
    return Int(info)


def lgels[
    T: DType
](
    t: Int8, m: Int, n: Int, k: Int, A: P[T], lda: Int32, B: P[T], ldb: Int32
) -> Int:
    comptime name = "dgels$NEWLAPACK" if T == DType.float64 else "sgels$NEWLAPACK"
    var t_ = t
    var m_ = Int32(m)
    var n_ = Int32(n)
    var k_ = Int32(k)
    var la = lda
    var lb = ldb
    var info = Int32(0)
    var q = Scalar[T](0)
    var lw = Int32(-1)
    external_call[name, NoneType](
        Pointer(to=t_),
        Pointer(to=m_),
        Pointer(to=n_),
        Pointer(to=k_),
        A,
        Pointer(to=la),
        B,
        Pointer(to=lb),
        Pointer(to=q),
        Pointer(to=lw),
        Pointer(to=info),
    )
    lw = Int32(max(Int(q), 1))
    var w = buffer[T](Int(lw))
    external_call[name, NoneType](
        Pointer(to=t_),
        Pointer(to=m_),
        Pointer(to=n_),
        Pointer(to=k_),
        A,
        Pointer(to=la),
        B,
        Pointer(to=lb),
        w,
        Pointer(to=lw),
        Pointer(to=info),
    )
    release(w)
    return Int(info)


def lsytrf[
    T: DType
](u: Int8, n: Int, A: P[T], lda: Int32, piv: P[DType.int32]) -> Int:
    comptime name = "dsytrf$NEWLAPACK" if T == DType.float64 else "ssytrf$NEWLAPACK"
    var u_ = u
    var n_ = Int32(n)
    var ld = lda
    var info = Int32(0)
    var q = Scalar[T](0)
    var lw = Int32(-1)
    external_call[name, NoneType](
        Pointer(to=u_),
        Pointer(to=n_),
        A,
        Pointer(to=ld),
        piv,
        Pointer(to=q),
        Pointer(to=lw),
        Pointer(to=info),
    )
    lw = Int32(max(Int(q), 1))
    var w = buffer[T](Int(lw))
    external_call[name, NoneType](
        Pointer(to=u_),
        Pointer(to=n_),
        A,
        Pointer(to=ld),
        piv,
        w,
        Pointer(to=lw),
        Pointer(to=info),
    )
    release(w)
    return Int(info)


def lsytrs[
    T: DType
](
    u: Int8,
    n: Int,
    nrhs: Int,
    A: P[T],
    lda: Int32,
    piv: P[DType.int32],
    B: P[T],
    ldb: Int32,
) -> Int:
    comptime name = "dsytrs$NEWLAPACK" if T == DType.float64 else "ssytrs$NEWLAPACK"
    var u_ = u
    var n_ = Int32(n)
    var k_ = Int32(nrhs)
    var la = lda
    var lb = ldb
    var info = Int32(0)
    external_call[name, NoneType](
        Pointer(to=u_),
        Pointer(to=n_),
        Pointer(to=k_),
        A,
        Pointer(to=la),
        piv,
        B,
        Pointer(to=lb),
        Pointer(to=info),
    )
    return Int(info)


def lpotri[T: DType](u: Int32, n: Int, A: P[T], lda: Int32) -> Int:
    """The inverse from a Cholesky factor.  An upper factor is transposed
    into a lower one first (n^2 copies): Accelerate's multithreaded dlauum
    with UPLO = 'U', which dpotri calls, gives wrong results
    nondeterministically (macOS 26.6); with UPLO = 'L' it is correct."""
    if u == Lower:
        return lpotri_[T](u, n, A, lda)
    var ld = Int(lda)
    var l = buffer[T](max(n * n, 1))
    for j in range(n):
        for i in range(j + 1):
            l[unsafe_offset=j + i * n] = A[unsafe_offset=i + j * ld]
    var info = lpotri_[T](Lower, n, l, Int32(max(n, 1)))
    for j in range(n):
        for i in range(j + 1):
            A[unsafe_offset=i + j * ld] = l[unsafe_offset=j + i * n]
    release(l)
    return info


def pivots32[PT: DType](piv: P[PT], n: Int) -> P[DType.int32]:
    """The pivots as LAPACK's 32-bit integers: the array itself (int32) or
    a copy (released with release_pivots)."""
    comptime if PT == DType.int32:
        return piv.unsafe_bitcast[Int32]()
    else:
        var p = buffer[DType.int32](max(n, 1))
        for k in range(n):
            p[unsafe_offset=k] = Int32(piv[unsafe_offset=k])
        return p


def release_pivots[
    PT: DType
](p: P[DType.int32], piv: P[PT], n: Int, store: Bool):
    comptime if PT != DType.int32:
        if store:
            for k in range(n):
                piv[unsafe_offset=k] = Scalar[PT](p[unsafe_offset=k])
        release(p)


# ---- the routines of the generated code --------------------------------------


def gemm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[S],
    conjA: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    conjB: Bool,
    B: P[T],
    incRowB: Int,
    incColB: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    """C := alpha A B + beta C (conj ignored: real types)."""
    var a = orient(m, k, incRowA, incColA)
    var b = orient(k, n, incRowB, incColB)
    var c = orient(m, n, incRowC, incColC)
    if c[0]:
        cgemm[T](
            trans(a[0]),
            trans(b[0]),
            m,
            n,
            k,
            alpha.cast[T](),
            A,
            a[1],
            B,
            b[1],
            beta.cast[T](),
            C,
            c[1],
        )
    else:  # C^T = B^T A^T
        cgemm[T](
            flip(trans(b[0])),
            flip(trans(a[0])),
            n,
            m,
            k,
            alpha.cast[T](),
            B,
            b[1],
            A,
            a[1],
            beta.cast[T](),
            C,
            c[1],
        )


def gemmtr[
    T: DType, S: DType
](
    lower: Bool,
    n: Int,
    k: Int,
    alpha: Scalar[S],
    conjA: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    conjB: Bool,
    B: P[T],
    incRowB: Int,
    incColB: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    """The lower (or upper) triangle of the n x n matrix C := beta C +
    alpha A B: gemm on blocks of columns, about half the flops of gemm."""
    comptime nb = 128
    for j in range(0, n, nb):
        var jb = min(nb, n - j)
        if lower:  # C(j:n, j:j+jb)
            gemm(
                n - j,
                jb,
                k,
                alpha,
                False,
                A.unsafe_offset(j * incRowA),
                incRowA,
                incColA,
                False,
                B.unsafe_offset(j * incColB),
                incRowB,
                incColB,
                beta,
                C.unsafe_offset(j * incRowC + j * incColC),
                incRowC,
                incColC,
            )
        else:  # C(0:j+jb, j:j+jb)
            gemm(
                j + jb,
                jb,
                k,
                alpha,
                False,
                A,
                incRowA,
                incColA,
                False,
                B.unsafe_offset(j * incColB),
                incRowB,
                incColB,
                beta,
                C.unsafe_offset(j * incColC),
                incRowC,
                incColC,
            )


def gemv[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    x: P[T],
    incX: Int,
    beta: Scalar[S],
    y: P[T],
    incY: Int,
):
    """y := alpha A x + beta y."""
    var a = orient(m, n, incRowA, incColA)
    if a[0]:
        cgemv[T](
            NoTrans,
            m,
            n,
            alpha.cast[T](),
            A,
            a[1],
            x,
            incX,
            beta.cast[T](),
            y,
            incY,
        )
    else:
        cgemv[T](
            Trans,
            n,
            m,
            alpha.cast[T](),
            A,
            a[1],
            x,
            incX,
            beta.cast[T](),
            y,
            incY,
        )


def symm[
    T: DType, S: DType
](
    lower: Bool,
    m: Int,
    n: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    """C := alpha A B + beta C, A symmetric (lower or upper triangle)."""
    var a = orient(m, m, incRowA, incColA)
    var b = orient(m, n, incRowB, incColB)
    var c = orient(m, n, incRowC, incColC)
    var u = triangle(lower, m, incRowA, incColA)
    var Bp = B
    var ldb = b[1]
    var copied = b[0] != c[0]
    if copied:  # B in the orientation of C
        if c[0]:
            Bp = tocol(m, n, B, incRowB, incColB)
            ldb = Int32(max(m, 1))
        else:
            Bp = tocol(n, m, B, incColB, incRowB)
            ldb = Int32(max(n, 1))
    if c[0]:
        csymm[T](
            Left,
            u,
            m,
            n,
            alpha.cast[T](),
            A,
            a[1],
            Bp,
            ldb,
            beta.cast[T](),
            C,
            c[1],
        )
    else:  # C^T = B^T A
        csymm[T](
            Right,
            u,
            n,
            m,
            alpha.cast[T](),
            A,
            a[1],
            Bp,
            ldb,
            beta.cast[T](),
            C,
            c[1],
        )
    if copied:
        release(Bp)


def sylmm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    symm(
        True,
        m,
        n,
        alpha,
        A,
        incRowA,
        incColA,
        B,
        incRowB,
        incColB,
        beta,
        C,
        incRowC,
        incColC,
    )


def syumm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    symm(
        False,
        m,
        n,
        alpha,
        A,
        incRowA,
        incColA,
        B,
        incRowB,
        incColB,
        beta,
        C,
        incRowC,
        incColC,
    )


def sylmv[
    T: DType, S: DType
](
    n: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    x: P[T],
    incX: Int,
    beta: Scalar[S],
    y: P[T],
    incY: Int,
):
    """y := alpha A x + beta y, A symmetric, lower triangle."""
    var a = orient(n, n, incRowA, incColA)
    csymv[T](
        triangle(True, n, incRowA, incColA),
        n,
        alpha.cast[T](),
        A,
        a[1],
        x,
        incX,
        beta.cast[T](),
        y,
        incY,
    )


def syrk[
    T: DType, S: DType
](
    lower: Bool,
    n: Int,
    k: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    """C := alpha A A^T + beta C, A n x k, the lower or upper triangle of C."""
    var a = orient(n, k, incRowA, incColA)
    var c = orient(n, n, incRowC, incColC)
    csyrk[T](
        triangle(lower, n, incRowC, incColC),
        trans(a[0]),
        n,
        k,
        alpha.cast[T](),
        A,
        a[1],
        beta.cast[T](),
        C,
        c[1],
    )


def sylrk[
    T: DType, S: DType
](
    n: Int,
    k: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    syrk(True, n, k, alpha, A, incRowA, incColA, beta, C, incRowC, incColC)


def syurk[
    T: DType, S: DType
](
    n: Int,
    k: Int,
    alpha: Scalar[S],
    A: P[T],
    incRowA: Int,
    incColA: Int,
    beta: Scalar[S],
    C: P[T],
    incRowC: Int,
    incColC: Int,
):
    syrk(False, n, k, alpha, A, incRowA, incColA, beta, C, incRowC, incColC)


def trxm[
    T: DType, S: DType, solve: Bool
](
    lower: Bool,
    m: Int,
    n: Int,
    alpha: Scalar[S],
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    """B := alpha A^-1 B (solve) or alpha A B, A m x m triangular, B m x n."""
    var a = orient(m, m, incRowA, incColA)
    var b = orient(m, n, incRowB, incColB)
    var u = Lower if lower == a[0] else Upper  # the triangle of the array
    var d = Unit if unitDiag else NonUnit
    var al = alpha.cast[T]()
    if b[0]:
        comptime if solve:
            ctrsm[T](Left, u, trans(a[0]), d, m, n, al, A, a[1], B, b[1])
        else:
            ctrmm[T](Left, u, trans(a[0]), d, m, n, al, A, a[1], B, b[1])
    else:  # X^T op(A)^T = B^T
        comptime if solve:
            ctrsm[T](Right, u, flip(trans(a[0])), d, n, m, al, A, a[1], B, b[1])
        else:
            ctrmm[T](Right, u, flip(trans(a[0])), d, n, m, al, A, a[1], B, b[1])


def trlmm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    conj: Bool,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    trxm[T, S, False](
        True, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB
    )


def trumm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    conj: Bool,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    trxm[T, S, False](
        False, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB
    )


def trlsm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    conj: Bool,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    trxm[T, S, True](
        True, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB
    )


def trusm[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    conj: Bool,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    trxm[T, S, True](
        False, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB
    )


def trtrs[
    T: DType
](
    lower: Bool,
    trans_: Bool,
    conj: Bool,
    unitDiag: Bool,
    n: Int,
    nrhs: Int,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
) -> Int:
    """op(A) X = B (a zero on the diagonal is not checked: ulm.lam calls it
    after a successful factorisation)."""
    if trans_:
        trxm[T, T, True](
            not lower,
            n,
            nrhs,
            Scalar[T](1),
            unitDiag,
            A,
            incColA,
            incRowA,
            B,
            incRowB,
            incColB,
        )
    else:
        trxm[T, T, True](
            lower,
            n,
            nrhs,
            Scalar[T](1),
            unitDiag,
            A,
            incRowA,
            incColA,
            B,
            incRowB,
            incColB,
        )
    return 0


def trsv[
    T: DType
](
    lower: Bool,
    n: Int,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    x: P[T],
    incX: Int,
):
    """x := A^-1 x, A lower or upper triangular."""
    var a = orient(n, n, incRowA, incColA)
    var u = Lower if lower == a[0] else Upper
    ctrsv[T](u, trans(a[0]), Unit if unitDiag else NonUnit, n, A, a[1], x, incX)


def trlsv[
    T: DType
](
    n: Int,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    x: P[T],
    incX: Int,
):
    trsv(True, n, unitDiag, A, incRowA, incColA, x, incX)


def trusv[
    T: DType
](
    n: Int,
    unitDiag: Bool,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    x: P[T],
    incX: Int,
):
    trsv(False, n, unitDiag, A, incRowA, incColA, x, incX)


def geaxpy[
    T: DType, S: DType
](
    m: Int,
    n: Int,
    alpha: Scalar[S],
    X: P[T],
    incRowX: Int,
    incColX: Int,
    Y: P[T],
    incRowY: Int,
    incColY: Int,
):
    """Y := alpha X + Y (no BLAS routine for strided matrices)."""
    var a = alpha.cast[T]()
    if incColY == 1:  # the inner loop along the unit stride of Y
        for i in range(m):
            for j in range(n):
                var k = i * incRowY + j
                Y[unsafe_offset=k] = (
                    Y[unsafe_offset=k]
                    + a * X[unsafe_offset=i * incRowX + j * incColX]
                )
    else:
        for j in range(n):
            for i in range(m):
                var k = i * incRowY + j * incColY
                Y[unsafe_offset=k] = (
                    Y[unsafe_offset=k]
                    + a * X[unsafe_offset=i * incRowX + j * incColX]
                )


def gescal[
    T: DType, S: DType
](m: Int, n: Int, alpha: Scalar[S], X: P[T], incRowX: Int, incColX: Int):
    """X := alpha X; as BLAS's scal, alpha = 0 sets X to zero (also NaN and
    Inf)."""
    var a = alpha.cast[T]()
    for i in range(m):
        for j in range(n):
            var k = i * incRowX + j * incColX
            X[unsafe_offset=k] = (
                Scalar[T](0) if a == 0 else X[unsafe_offset=k] * a
            )


def potrf[
    T: DType
](lower: Bool, n: Int, A: P[T], incRowA: Int, incColA: Int) -> Int:
    """A = L L^T (lower) or U^T U, in place; returns LAPACK's info."""
    var a = orient(n, n, incRowA, incColA)
    return lpotrf[T](triangle(lower, n, incRowA, incColA), n, A, a[1])


def potri[
    T: DType
](lower: Bool, n: Int, A: P[T], incRowA: Int, incColA: Int) -> Int:
    """The inverse from the Cholesky factor of potrf (same uplo and
    strides)."""
    var a = orient(n, n, incRowA, incColA)
    return lpotri[T](triangle(lower, n, incRowA, incColA), n, A, a[1])


def getrf[
    T: DType, PT: DType
](
    m: Int, n: Int, A: P[T], incRowA: Int, incColA: Int, piv: P[PT], incPiv: Int
) -> Int:
    """P A = L U in place; a row-major A is factorised as the column-major
    A^T.  The pivots are LAPACK's (1-based), in the IR's integer type."""
    var a = orient(m, n, incRowA, incColA)
    var k = min(m, n)
    var p = pivots32(piv, k)
    var info = lgetrf[T](m, n, A, a[1], p) if a[0] else lgetrf[T](
        n, m, A, a[1], p
    )
    release_pivots(p, piv, k, True)
    return info


def getrs[
    T: DType, PT: DType
](
    trans_: Bool,
    conj: Bool,
    n: Int,
    nrhs: Int,
    LU: P[T],
    incRowA: Int,
    incColA: Int,
    piv: P[PT],
    incPiv: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    """B := op(A)^-1 B with the factors of getrf."""
    var a = orient(n, n, incRowA, incColA)
    var b = orient(n, nrhs, incRowB, incColB)
    # the factors of A (column-major) or of A^T
    var t = char("T") if trans_ == a[0] else char("N")
    var p = pivots32(piv, n)
    if b[0]:
        _ = lgetrs[T](t, n, nrhs, LU, a[1], p, B, b[1])
    else:
        var c = tocol(n, nrhs, B, incRowB, incColB)
        _ = lgetrs[T](t, n, nrhs, LU, a[1], p, c, Int32(max(n, 1)))
        fromcol(B, c, n, nrhs, incRowB, incColB)
    release_pivots(p, piv, n, False)


def getri[
    T: DType, PT: DType
](n: Int, A: P[T], incRowA: Int, incColA: Int, piv: P[PT], incPiv: Int) -> Int:
    """A^-1 from the factors of getrf (same strides), in place."""
    var a = orient(n, n, incRowA, incColA)
    var p = pivots32(piv, n)
    var info = lgetri[T](n, A, a[1], p)
    release_pivots(p, piv, n, False)
    return info


def gels[
    T: DType
](
    m: Int,
    n: Int,
    k: Int,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
) -> Int:
    """Least squares by QR (orthogonal-triangular factorisation): B
    (m x k) := the minimiser of ||A X - B|| in its leading n rows; a
    row-major A is the column-major A^T (gels with 'T')."""
    var a = orient(m, n, incRowA, incColA)
    var c = tocol(m, k, B, incRowB, incColB)
    var info = lgels[T](
        char("N") if a[0] else char("T"),
        m if a[0] else n,
        n if a[0] else m,
        k,
        A,
        a[1],
        c,
        Int32(max(m, 1)),
    )
    fromcol(B, c, m, k, incRowB, incColB)
    return info


def sytrf[
    T: DType, PT: DType
](
    m: Int, n: Int, A: P[T], incRowA: Int, incColA: Int, piv: P[PT], incPiv: Int
) -> Int:
    """A = L D L^T (Bunch-Kaufman), in place (m = n, the rows)."""
    var a = orient(n, n, incRowA, incColA)
    var p = pivots32(piv, n)
    var info = lsytrf[T](char("L") if a[0] else char("U"), n, A, a[1], p)
    release_pivots(p, piv, n, True)
    return info


def sytrs[
    T: DType, PT: DType
](
    lower: Bool,
    conj: Bool,
    n: Int,
    nrhs: Int,
    LDL: P[T],
    incRowA: Int,
    incColA: Int,
    piv: P[PT],
    incPiv: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    """B := A^-1 B with the factors of sytrf."""
    var a = orient(n, n, incRowA, incColA)
    var b = orient(n, nrhs, incRowB, incColB)
    var u = char("L") if a[0] else char("U")
    var p = pivots32(piv, n)
    if b[0]:
        _ = lsytrs[T](u, n, nrhs, LDL, a[1], p, B, b[1])
    else:
        var c = tocol(n, nrhs, B, incRowB, incColB)
        _ = lsytrs[T](u, n, nrhs, LDL, a[1], p, c, Int32(max(n, 1)))
        fromcol(B, c, n, nrhs, incRowB, incColB)
    release_pivots(p, piv, n, False)


# ---- the helpers the lowering emits as C++ (ulm_lam_*) -----------------------


def copy[T: DType, U: DType](dst: P[T], src: P[U], bytes: Int):
    """memref.copy."""
    _ = external_call["memcpy", Int](dst, src, bytes)


def gecopy[
    T: DType
](
    m: Int,
    n: Int,
    A: P[T],
    incRowA: Int,
    incColA: Int,
    B: P[T],
    incRowB: Int,
    incColB: Int,
):
    """B := op(A) (ulm_lamk.gecopy), blocked."""
    comptime bs = 64
    for i0 in range(0, m, bs):
        for j0 in range(0, n, bs):
            for i in range(i0, min(i0 + bs, m)):
                for j in range(j0, min(j0 + bs, n)):
                    B[unsafe_offset=i * incRowB + j * incColB] = A[
                        unsafe_offset=i * incRowA + j * incColA
                    ]


def adddiag[
    T: DType, S: DType
](n: Int, alpha: Scalar[S], A: P[T], incRow: Int, incCol: Int):
    """A := A + alpha I."""
    for i in range(n):
        var k = i * (incRow + incCol)
        A[unsafe_offset=k] = A[unsafe_offset=k] + alpha.cast[T]()


def sycomplete[
    T: DType
](lower: Bool, n: Int, A: P[T], incRow: Int, incCol: Int):
    """Mirror the stored triangle of a symmetric matrix, in tiles."""
    comptime b = 64
    for ib in range(0, n, b):
        for jb in range(0, ib + 1, b):
            for i in range(ib, min(n, ib + b)):
                for j in range(jb, min(i, jb + b)):
                    var lo = i * incRow + j * incCol
                    var up = j * incRow + i * incCol
                    if lower:
                        A[unsafe_offset=up] = A[unsafe_offset=lo]
                    else:
                        A[unsafe_offset=lo] = A[unsafe_offset=up]


# ---- C's integer division (truncating; Mojo's // and % floor) ----------------


def cdiv(a: Int, b: Int) -> Int:
    var q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


def crem(a: Int, b: Int) -> Int:
    return a - cdiv(a, b) * b
