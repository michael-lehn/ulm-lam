// cblas_backend.h - the ulmBLAS interface used by ulm.lam, implemented with
// CBLAS and LAPACK (backend=cblas of convert-ulm-lamk-to-ulmblas).
//
// ulm.lam's lowering calls ulmBLAS routines on strided views (incRow,
// incCol).  With backend=cblas the generated code calls the routines of
// the namespace ulm_lam_cblas instead, with the same arguments, and this
// header maps each call onto CBLAS (Basic Linear Algebra Subprograms, C
// interface) and LAPACK (Linear Algebra PACKage, Fortran symbols), as MKL
// (Intel Math Kernel Library), OpenBLAS and Apple's Accelerate provide
// them.  So the same kernel sequence runs on any BLAS: the mapping of
// ulm.lam and the BLAS implementation can be measured separately.
//
// No copies of matrices: every operand of the generated code is a
// row-major or column-major array (one stride is 1), and a row-major
// matrix is the transpose of a column-major one.  The adapter works in
// column-major order and turns the other orientation into a trans flag, a
// flipped triangle, or the transposed problem (C^T = B^T A^T).  Only a
// right-hand side of getrs and the B of symm whose orientation differs
// from that of C are copied (O(n m), not O(n^2)).
//
// The BLAS integer is int (LP64).  The declarations are our own (the
// CBLAS enums are ints), so no vendor header is needed; link with e.g.
// -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core, -lopenblas, or
// -framework Accelerate.

#ifndef ULM_LAM_CBLAS_BACKEND_H
#define ULM_LAM_CBLAS_BACKEND_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

// ULM_LAM_CBLAS_ILP64: 64-bit integers and the symbol suffix 64_, as
// libblastrampoline exports them (Julia's BLAS: OpenBLAS by default, or
// whatever it forwards to, e.g. ulmBLAS via UlmBLAS.forward()).
#ifdef ULM_LAM_CBLAS_ILP64
typedef std::int64_t ulm_lam_blasint;
#define ULM_LAM_SYM(f) f##64_
#else
typedef int ulm_lam_blasint;
#define ULM_LAM_SYM(f) f
#endif

extern "C" {
// CBLAS
void ULM_LAM_SYM(cblas_dgemm)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              ulm_lam_blasint, double, const double *,
                              ulm_lam_blasint, const double *, ulm_lam_blasint,
                              double, double *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_sgemm)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              ulm_lam_blasint, float, const float *,
                              ulm_lam_blasint, const float *, ulm_lam_blasint,
                              float, float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dgemv)(int, int, ulm_lam_blasint, ulm_lam_blasint,
                              double, const double *, ulm_lam_blasint,
                              const double *, ulm_lam_blasint, double, double *,
                              ulm_lam_blasint);
void ULM_LAM_SYM(cblas_sgemv)(int, int, ulm_lam_blasint, ulm_lam_blasint, float,
                              const float *, ulm_lam_blasint, const float *,
                              ulm_lam_blasint, float, float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dsymm)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              double, const double *, ulm_lam_blasint,
                              const double *, ulm_lam_blasint, double, double *,
                              ulm_lam_blasint);
void ULM_LAM_SYM(cblas_ssymm)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              float, const float *, ulm_lam_blasint,
                              const float *, ulm_lam_blasint, float, float *,
                              ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dsymv)(int, int, ulm_lam_blasint, double, const double *,
                              ulm_lam_blasint, const double *, ulm_lam_blasint,
                              double, double *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_ssymv)(int, int, ulm_lam_blasint, float, const float *,
                              ulm_lam_blasint, const float *, ulm_lam_blasint,
                              float, float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dsyrk)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              double, const double *, ulm_lam_blasint, double,
                              double *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_ssyrk)(int, int, int, ulm_lam_blasint, ulm_lam_blasint,
                              float, const float *, ulm_lam_blasint, float,
                              float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dtrmm)(int, int, int, int, int, ulm_lam_blasint,
                              ulm_lam_blasint, double, const double *,
                              ulm_lam_blasint, double *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_strmm)(int, int, int, int, int, ulm_lam_blasint,
                              ulm_lam_blasint, float, const float *,
                              ulm_lam_blasint, float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dtrsm)(int, int, int, int, int, ulm_lam_blasint,
                              ulm_lam_blasint, double, const double *,
                              ulm_lam_blasint, double *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_strsm)(int, int, int, int, int, ulm_lam_blasint,
                              ulm_lam_blasint, float, const float *,
                              ulm_lam_blasint, float *, ulm_lam_blasint);
void ULM_LAM_SYM(cblas_dtrsv)(int, int, int, int, ulm_lam_blasint,
                              const double *, ulm_lam_blasint, double *,
                              ulm_lam_blasint);
void ULM_LAM_SYM(cblas_strsv)(int, int, int, int, ulm_lam_blasint,
                              const float *, ulm_lam_blasint, float *,
                              ulm_lam_blasint);
// LAPACK (Fortran; trailing hidden lengths of the character arguments)
void ULM_LAM_SYM(dpotrf_)(const char *, const ulm_lam_blasint *, double *,
                          const ulm_lam_blasint *, ulm_lam_blasint *,
                          std::size_t);
void ULM_LAM_SYM(spotrf_)(const char *, const ulm_lam_blasint *, float *,
                          const ulm_lam_blasint *, ulm_lam_blasint *,
                          std::size_t);
void ULM_LAM_SYM(dgetrf_)(const ulm_lam_blasint *, const ulm_lam_blasint *,
                          double *, const ulm_lam_blasint *, ulm_lam_blasint *,
                          ulm_lam_blasint *);
void ULM_LAM_SYM(sgetrf_)(const ulm_lam_blasint *, const ulm_lam_blasint *,
                          float *, const ulm_lam_blasint *, ulm_lam_blasint *,
                          ulm_lam_blasint *);
void ULM_LAM_SYM(dgetrs_)(const char *, const ulm_lam_blasint *,
                          const ulm_lam_blasint *, const double *,
                          const ulm_lam_blasint *, const ulm_lam_blasint *,
                          double *, const ulm_lam_blasint *, ulm_lam_blasint *,
                          std::size_t);
void ULM_LAM_SYM(sgetrs_)(const char *, const ulm_lam_blasint *,
                          const ulm_lam_blasint *, const float *,
                          const ulm_lam_blasint *, const ulm_lam_blasint *,
                          float *, const ulm_lam_blasint *, ulm_lam_blasint *,
                          std::size_t);
}

namespace ulm_lam_cblas {

namespace detail {

enum
{
    ColMajor = 102
};
enum
{
    NoTrans = 111,
    Trans = 112
};
enum
{
    Upper = 121,
    Lower = 122
};
enum
{
    NonUnit = 131,
    Unit = 132
};
enum
{
    Left = 141,
    Right = 142
};

inline int
flip(int t)
{
    return t == NoTrans ? Trans : NoTrans;
}

/// A rows x cols view with strides (incRow, incCol) as op(M) of a
/// column-major array M: col (op = NoTrans, M = the view) or not (op =
/// Trans, M = the transposed view), and M's leading dimension.
struct Op
{
    bool col;
    ulm_lam_blasint ld;
    int
    trans() const
    {
	return col ? NoTrans : Trans;
    }
};

template <typename I>
Op
orient(I rows, I cols, I incRow, I incCol)
{
    // (BLAS wants a leading dimension >= 1, also for empty matrices)
    if (incRow == 1 && (cols <= 1 || incCol >= rows))
	return {true,
	        static_cast<ulm_lam_blasint>(
	            cols <= 1 ? std::max<I>(rows, 1) : std::max<I>(incCol, 1))};
    if (incCol == 1)
	return {false,
	        static_cast<ulm_lam_blasint>(
	            rows <= 1 ? std::max<I>(cols, 1) : std::max<I>(incRow, 1))};
    std::abort(); // neither row- nor column-major: not generated by ulm.lam
}

// Overloads by element type.
inline void
gemm(ulm_lam_blasint o, ulm_lam_blasint ta, ulm_lam_blasint tb,
     ulm_lam_blasint m, ulm_lam_blasint n, ulm_lam_blasint k, double al,
     const double *A, ulm_lam_blasint lda, const double *B, ulm_lam_blasint ldb,
     double be, double *C, ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_dgemm)(o, ta, tb, m, n, k, al, A, lda, B, ldb, be, C,
                             ldc);
}
inline void
gemm(ulm_lam_blasint o, ulm_lam_blasint ta, ulm_lam_blasint tb,
     ulm_lam_blasint m, ulm_lam_blasint n, ulm_lam_blasint k, float al,
     const float *A, ulm_lam_blasint lda, const float *B, ulm_lam_blasint ldb,
     float be, float *C, ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_sgemm)(o, ta, tb, m, n, k, al, A, lda, B, ldb, be, C,
                             ldc);
}
inline void
gemv(ulm_lam_blasint o, ulm_lam_blasint t, ulm_lam_blasint m, ulm_lam_blasint n,
     double al, const double *A, ulm_lam_blasint lda, const double *x,
     ulm_lam_blasint ix, double be, double *y, ulm_lam_blasint iy)
{
    ULM_LAM_SYM(cblas_dgemv)(o, t, m, n, al, A, lda, x, ix, be, y, iy);
}
inline void
gemv(ulm_lam_blasint o, ulm_lam_blasint t, ulm_lam_blasint m, ulm_lam_blasint n,
     float al, const float *A, ulm_lam_blasint lda, const float *x,
     ulm_lam_blasint ix, float be, float *y, ulm_lam_blasint iy)
{
    ULM_LAM_SYM(cblas_sgemv)(o, t, m, n, al, A, lda, x, ix, be, y, iy);
}
inline void
symm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint m,
     ulm_lam_blasint n, double al, const double *A, ulm_lam_blasint lda,
     const double *B, ulm_lam_blasint ldb, double be, double *C,
     ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_dsymm)(o, s, u, m, n, al, A, lda, B, ldb, be, C, ldc);
}
inline void
symm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint m,
     ulm_lam_blasint n, float al, const float *A, ulm_lam_blasint lda,
     const float *B, ulm_lam_blasint ldb, float be, float *C,
     ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_ssymm)(o, s, u, m, n, al, A, lda, B, ldb, be, C, ldc);
}
inline void
symv(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint n, double al,
     const double *A, ulm_lam_blasint lda, const double *x, ulm_lam_blasint ix,
     double be, double *y, ulm_lam_blasint iy)
{
    ULM_LAM_SYM(cblas_dsymv)(o, u, n, al, A, lda, x, ix, be, y, iy);
}
inline void
symv(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint n, float al,
     const float *A, ulm_lam_blasint lda, const float *x, ulm_lam_blasint ix,
     float be, float *y, ulm_lam_blasint iy)
{
    ULM_LAM_SYM(cblas_ssymv)(o, u, n, al, A, lda, x, ix, be, y, iy);
}
inline void
syrk(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint t, ulm_lam_blasint n,
     ulm_lam_blasint k, double al, const double *A, ulm_lam_blasint lda,
     double be, double *C, ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_dsyrk)(o, u, t, n, k, al, A, lda, be, C, ldc);
}
inline void
syrk(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint t, ulm_lam_blasint n,
     ulm_lam_blasint k, float al, const float *A, ulm_lam_blasint lda, float be,
     float *C, ulm_lam_blasint ldc)
{
    ULM_LAM_SYM(cblas_ssyrk)(o, u, t, n, k, al, A, lda, be, C, ldc);
}
inline void
trmm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint t,
     ulm_lam_blasint d, ulm_lam_blasint m, ulm_lam_blasint n, double al,
     const double *A, ulm_lam_blasint lda, double *B, ulm_lam_blasint ldb)
{
    ULM_LAM_SYM(cblas_dtrmm)(o, s, u, t, d, m, n, al, A, lda, B, ldb);
}
inline void
trmm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint t,
     ulm_lam_blasint d, ulm_lam_blasint m, ulm_lam_blasint n, float al,
     const float *A, ulm_lam_blasint lda, float *B, ulm_lam_blasint ldb)
{
    ULM_LAM_SYM(cblas_strmm)(o, s, u, t, d, m, n, al, A, lda, B, ldb);
}
inline void
trsm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint t,
     ulm_lam_blasint d, ulm_lam_blasint m, ulm_lam_blasint n, double al,
     const double *A, ulm_lam_blasint lda, double *B, ulm_lam_blasint ldb)
{
    ULM_LAM_SYM(cblas_dtrsm)(o, s, u, t, d, m, n, al, A, lda, B, ldb);
}
inline void
trsm(ulm_lam_blasint o, ulm_lam_blasint s, ulm_lam_blasint u, ulm_lam_blasint t,
     ulm_lam_blasint d, ulm_lam_blasint m, ulm_lam_blasint n, float al,
     const float *A, ulm_lam_blasint lda, float *B, ulm_lam_blasint ldb)
{
    ULM_LAM_SYM(cblas_strsm)(o, s, u, t, d, m, n, al, A, lda, B, ldb);
}
inline void
trsv(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint t, ulm_lam_blasint d,
     ulm_lam_blasint n, const double *A, ulm_lam_blasint lda, double *x,
     ulm_lam_blasint ix)
{
    ULM_LAM_SYM(cblas_dtrsv)(o, u, t, d, n, A, lda, x, ix);
}
inline void
trsv(ulm_lam_blasint o, ulm_lam_blasint u, ulm_lam_blasint t, ulm_lam_blasint d,
     ulm_lam_blasint n, const float *A, ulm_lam_blasint lda, float *x,
     ulm_lam_blasint ix)
{
    ULM_LAM_SYM(cblas_strsv)(o, u, t, d, n, A, lda, x, ix);
}
inline ulm_lam_blasint
potrf(char u, ulm_lam_blasint n, double *A, ulm_lam_blasint lda)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(dpotrf_)(&u, &n, A, &lda, &info, 1);
    return info;
}
inline ulm_lam_blasint
potrf(char u, ulm_lam_blasint n, float *A, ulm_lam_blasint lda)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(spotrf_)(&u, &n, A, &lda, &info, 1);
    return info;
}
inline ulm_lam_blasint
getrf(ulm_lam_blasint m, ulm_lam_blasint n, double *A, ulm_lam_blasint lda,
      ulm_lam_blasint *piv)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(dgetrf_)(&m, &n, A, &lda, piv, &info);
    return info;
}
inline ulm_lam_blasint
getrf(ulm_lam_blasint m, ulm_lam_blasint n, float *A, ulm_lam_blasint lda,
      ulm_lam_blasint *piv)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(sgetrf_)(&m, &n, A, &lda, piv, &info);
    return info;
}
inline void
getrs(char t, ulm_lam_blasint n, ulm_lam_blasint nrhs, const double *A,
      ulm_lam_blasint lda, const ulm_lam_blasint *piv, double *B,
      ulm_lam_blasint ldb)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(dgetrs_)(&t, &n, &nrhs, A, &lda, piv, B, &ldb, &info, 1);
}
inline void
getrs(char t, ulm_lam_blasint n, ulm_lam_blasint nrhs, const float *A,
      ulm_lam_blasint lda, const ulm_lam_blasint *piv, float *B,
      ulm_lam_blasint ldb)
{
    ulm_lam_blasint info;
    ULM_LAM_SYM(sgetrs_)(&t, &n, &nrhs, A, &lda, piv, B, &ldb, &info, 1);
}

/// B (rows x cols, strides incRow/incCol) copied into / from a column-major
/// array.
template <typename I, typename T>
std::vector<T>
toCol(I rows, I cols, const T *B, I incRow, I incCol)
{
    std::vector<T> c(static_cast<std::size_t>(rows) * cols);
    for (I j = 0; j < cols; ++j)
	for (I i = 0; i < rows; ++i)
	    c[j * rows + i] = B[i * incRow + j * incCol];
    return c;
}

template <typename I, typename T>
void
fromCol(I rows, I cols, const std::vector<T> &c, T *B, I incRow, I incCol)
{
    for (I j = 0; j < cols; ++j)
	for (I i = 0; i < rows; ++i)
	    B[i * incRow + j * incCol] = c[j * rows + i];
}

} // namespace detail

// C := alpha A B + beta C (A, B as views; conj ignored: real types)
template <typename I, typename T>
void
gemm(I m, I n, I k, T alpha, bool, const T *A, I incRowA, I incColA, bool,
     const T *B, I incRowB, I incColB, T beta, T *C, I incRowC, I incColC)
{
    using namespace detail;
    Op a = orient(m, k, incRowA, incColA), b = orient(k, n, incRowB, incColB);
    Op c = orient(m, n, incRowC, incColC);
    if (c.col)
	detail::gemm(ColMajor, a.trans(), b.trans(), m, n, k, alpha, A, a.ld, B,
	             b.ld, beta, C, c.ld);
    else // C^T = B^T A^T
	detail::gemm(ColMajor, flip(b.trans()), flip(a.trans()), n, m, k, alpha,
	             B, b.ld, A, a.ld, beta, C, c.ld);
}

// y := alpha A x + beta y
template <typename I, typename T>
void
gemv(I m, I n, T alpha, const T *A, I incRowA, I incColA, const T *x, I incX,
     T beta, T *y, I incY)
{
    using namespace detail;
    Op a = orient(m, n, incRowA, incColA);
    if (a.col)
	detail::gemv(ColMajor, NoTrans, m, n, alpha, A, a.ld, x, incX, beta, y,
	             incY);
    else
	detail::gemv(ColMajor, Trans, n, m, alpha, A, a.ld, x, incX, beta, y,
	             incY);
}

/// Triangle of a symmetric view in the column-major array: a row-major
/// view stores its lower triangle in the array's upper one.
template <typename I>
int
triangle(bool lower, I n, I incRow, I incCol)
{
    bool col = detail::orient(n, n, incRow, incCol).col;
    return lower == col ? detail::Lower : detail::Upper;
}

// C := alpha A B + beta C, A symmetric (lower or upper triangle)
template <typename I, typename T>
void
symm(bool lower, I m, I n, T alpha, const T *A, I incRowA, I incColA,
     const T *B, I incRowB, I incColB, T beta, T *C, I incRowC, I incColC)
{
    using namespace detail;
    Op a = orient(m, m, incRowA, incColA), b = orient(m, n, incRowB, incColB);
    Op c = orient(m, n, incRowC, incColC);
    int u = triangle(lower, m, incRowA, incColA);
    std::vector<T> tmp;
    if (b.col != c.col) { // B in the orientation of C
	tmp = toCol(c.col ? m : n, c.col ? n : m, B, c.col ? incRowB : incColB,
	            c.col ? incColB : incRowB);
	B = tmp.data();
	b.ld = static_cast<ulm_lam_blasint>(c.col ? std::max<I>(m, 1)
	                                          : std::max<I>(n, 1));
    }
    if (c.col)
	detail::symm(ColMajor, Left, u, m, n, alpha, A, a.ld, B, b.ld, beta, C,
	             c.ld);
    else // C^T = B^T A
	detail::symm(ColMajor, Right, u, n, m, alpha, A, a.ld, B, b.ld, beta, C,
	             c.ld);
}

template <typename I, typename T>
void
sylmm(I m, I n, T alpha, const T *A, I incRowA, I incColA, const T *B,
      I incRowB, I incColB, T beta, T *C, I incRowC, I incColC)
{
    symm(true, m, n, alpha, A, incRowA, incColA, B, incRowB, incColB, beta, C,
         incRowC, incColC);
}

template <typename I, typename T>
void
syumm(I m, I n, T alpha, const T *A, I incRowA, I incColA, const T *B,
      I incRowB, I incColB, T beta, T *C, I incRowC, I incColC)
{
    symm(false, m, n, alpha, A, incRowA, incColA, B, incRowB, incColB, beta, C,
         incRowC, incColC);
}

// y := alpha A x + beta y, A symmetric, lower triangle
template <typename I, typename T>
void
sylmv(I n, T alpha, const T *A, I incRowA, I incColA, const T *x, I incX,
      T beta, T *y, I incY)
{
    using namespace detail;
    Op a = orient(n, n, incRowA, incColA);
    detail::symv(ColMajor, triangle(true, n, incRowA, incColA), n, alpha, A,
                 a.ld, x, incX, beta, y, incY);
}

// C := alpha A A^T + beta C, A n x k, the lower or upper triangle of C
template <typename I, typename T>
void
syrk(bool lower, I n, I k, T alpha, const T *A, I incRowA, I incColA, T beta,
     T *C, I incRowC, I incColC)
{
    using namespace detail;
    Op a = orient(n, k, incRowA, incColA), c = orient(n, n, incRowC, incColC);
    detail::syrk(ColMajor, triangle(lower, n, incRowC, incColC), a.trans(), n,
                 k, alpha, A, a.ld, beta, C, c.ld);
}

template <typename I, typename T>
void
sylrk(I n, I k, T alpha, const T *A, I incRowA, I incColA, T beta, T *C,
      I incRowC, I incColC)
{
    syrk(true, n, k, alpha, A, incRowA, incColA, beta, C, incRowC, incColC);
}

template <typename I, typename T>
void
syurk(I n, I k, T alpha, const T *A, I incRowA, I incColA, T beta, T *C,
      I incRowC, I incColC)
{
    syrk(false, n, k, alpha, A, incRowA, incColA, beta, C, incRowC, incColC);
}

// B := alpha A^-1 B, A m x m lower or upper triangular, B m x n
template <typename I, typename T>
void
trsm(bool lower, I m, I n, T alpha, bool unitDiag, const T *A, I incRowA,
     I incColA, T *B, I incRowB, I incColB)
{
    using namespace detail;
    Op a = orient(m, m, incRowA, incColA), b = orient(m, n, incRowB, incColB);
    int u = (lower == a.col) ? Lower : Upper; // triangle of the array
    int d = unitDiag ? Unit : NonUnit;
    if (b.col)
	detail::trsm(ColMajor, Left, u, a.trans(), d, m, n, alpha, A, a.ld, B,
	             b.ld);
    else // X^T op(A)^T = B^T
	detail::trsm(ColMajor, Right, u, flip(a.trans()), d, n, m, alpha, A,
	             a.ld, B, b.ld);
}

// B := alpha A B, A m x m lower or upper triangular, B m x n
template <typename I, typename T>
void
trmm(bool lower, I m, I n, T alpha, bool unitDiag, const T *A, I incRowA,
     I incColA, T *B, I incRowB, I incColB)
{
    using namespace detail;
    Op a = orient(m, m, incRowA, incColA), b = orient(m, n, incRowB, incColB);
    int u = (lower == a.col) ? Lower : Upper; // triangle of the array
    int d = unitDiag ? Unit : NonUnit;
    if (b.col)
	detail::trmm(ColMajor, Left, u, a.trans(), d, m, n, alpha, A, a.ld, B,
	             b.ld);
    else // B^T := B^T op(A)^T
	detail::trmm(ColMajor, Right, u, flip(a.trans()), d, n, m, alpha, A,
	             a.ld, B, b.ld);
}

template <typename I, typename T>
void
trlmm(I m, I n, T alpha, bool, bool unitDiag, const T *A, I incRowA, I incColA,
      T *B, I incRowB, I incColB)
{
    trmm(true, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB);
}

template <typename I, typename T>
void
trumm(I m, I n, T alpha, bool, bool unitDiag, const T *A, I incRowA, I incColA,
      T *B, I incRowB, I incColB)
{
    trmm(false, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB,
         incColB);
}

template <typename I, typename T>
void
trlsm(I m, I n, T alpha, bool, bool unitDiag, const T *A, I incRowA, I incColA,
      T *B, I incRowB, I incColB)
{
    trsm(true, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB, incColB);
}

template <typename I, typename T>
void
trusm(I m, I n, T alpha, bool, bool unitDiag, const T *A, I incRowA, I incColA,
      T *B, I incRowB, I incColB)
{
    trsm(false, m, n, alpha, unitDiag, A, incRowA, incColA, B, incRowB,
         incColB);
}

// op(A) X = B (as ulmBLAS::trtrs; a zero on the diagonal is not checked:
// ulm.lam calls it after a successful factorisation)
template <typename I, typename T>
I
trtrs(bool lower, bool trans, bool, bool unitDiag, I n, I nrhs, const T *A,
      I incRowA, I incColA, T *B, I incRowB, I incColB)
{
    if (trans)
	trsm(!lower, n, nrhs, T(1), unitDiag, A, incColA, incRowA, B, incRowB,
	     incColB);
    else
	trsm(lower, n, nrhs, T(1), unitDiag, A, incRowA, incColA, B, incRowB,
	     incColB);
    return 0;
}

// x := A^-1 x, A lower or upper triangular
template <typename I, typename T>
void
trsv(bool lower, I n, bool unitDiag, const T *A, I incRowA, I incColA, T *x,
     I incX)
{
    using namespace detail;
    Op a = orient(n, n, incRowA, incColA);
    int u = (lower == a.col) ? Lower : Upper;
    detail::trsv(ColMajor, u, a.trans(), unitDiag ? Unit : NonUnit, n, A, a.ld,
                 x, incX);
}

template <typename I, typename T>
void
trlsv(I n, bool unitDiag, const T *A, I incRowA, I incColA, T *x, I incX)
{
    trsv(true, n, unitDiag, A, incRowA, incColA, x, incX);
}

template <typename I, typename T>
void
trusv(I n, bool unitDiag, const T *A, I incRowA, I incColA, T *x, I incX)
{
    trsv(false, n, unitDiag, A, incRowA, incColA, x, incX);
}

// Y := alpha X + Y, X := alpha X (no BLAS routine for strided matrices)
template <typename I, typename T>
void
geaxpy(I m, I n, T alpha, const T *X, I incRowX, I incColX, T *Y, I incRowY,
       I incColY)
{
    for (I i = 0; i < m; ++i)
	for (I j = 0; j < n; ++j)
	    Y[i * incRowY + j * incColY] +=
	        alpha * X[i * incRowX + j * incColX];
}

template <typename I, typename T>
void
gescal(I m, I n, T alpha, T *X, I incRowX, I incColX)
{
    // as BLAS's scal: alpha = 0 sets X to zero (also NaN and Inf)
    for (I i = 0; i < m; ++i)
	for (I j = 0; j < n; ++j)
	    X[i * incRowX + j * incColX] =
	        alpha == T(0) ? T(0) : X[i * incRowX + j * incColX] * alpha;
}

// A = L L^T (lower) or U^T U, in place; returns LAPACK's info
template <typename I, typename T>
I
potrf(bool lower, I n, T *A, I incRowA, I incColA)
{
    using namespace detail;
    Op a = orient(n, n, incRowA, incColA);
    bool lo = triangle(lower, n, incRowA, incColA) == Lower;
    return detail::potrf(lo ? 'L' : 'U', static_cast<ulm_lam_blasint>(n), A,
                         a.ld);
}

// P A = L U in place.  A row-major A is factorised as the column-major
// A^T; getrs, called with the same strides, solves with it accordingly.
// The pivots are LAPACK's (1-based, int), only read by getrs.
template <typename I, typename T, typename P>
I
getrf(I m, I n, T *A, I incRowA, I incColA, P *piv, I)
{
    using namespace detail;
    Op a = orient(m, n, incRowA, incColA);
    // pivots: the IR's integers, LAPACK's (64 bit with ILP64) in between
    std::vector<ulm_lam_blasint> p(std::min(m, n));
    I info = a.col ? detail::getrf(m, n, A, a.ld, p.data())
                   : detail::getrf(n, m, A, a.ld, p.data());
    for (std::size_t k = 0; k < p.size(); ++k)
	piv[k] = static_cast<P>(p[k]);
    return info;
}

// B := op(A)^-1 B with the factors of getrf
template <typename I, typename T, typename P>
void
getrs(bool trans, bool, I n, I nrhs, const T *LU, I incRowA, I incColA,
      const P *piv, I, T *B, I incRowB, I incColB)
{
    using namespace detail;
    Op a = orient(n, n, incRowA, incColA),
       b = orient(n, nrhs, incRowB, incColB);
    // factors of A (column-major) or of A^T (row-major)
    char t = (trans == a.col) ? 'T' : 'N';
    std::vector<ulm_lam_blasint> pv(piv, piv + n);
    const ulm_lam_blasint *p = pv.data();
    if (b.col) {
	detail::getrs(t, n, nrhs, LU, a.ld, p, B, b.ld);
	return;
    }
    std::vector<T> c = toCol(n, nrhs, B, incRowB, incColB);
    detail::getrs(t, n, nrhs, LU, a.ld, p, c.data(),
                  static_cast<ulm_lam_blasint>(std::max<I>(n, 1)));
    fromCol(n, nrhs, c, B, incRowB, incColB);
}

} // namespace ulm_lam_cblas

#endif // ULM_LAM_CBLAS_BACKEND_H
