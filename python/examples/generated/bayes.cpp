#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/trtrs.h>
#include <ulmblas/ulmblas.h>
static void
ulm_lam_copy(void *dst, const void *src, std::size_t bytes)
{
#ifdef _OPENMP
    const std::size_t chunk = std::size_t(1) << 20;
    if (bytes >= 8 * chunk) {
	std::ptrdiff_t chunks = (bytes + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
	for (std::ptrdiff_t c = 0; c < chunks; ++c) {
	    std::size_t off = c * chunk;
	    std::size_t len = bytes - off < chunk ? bytes - off : chunk;
	    std::memcpy(static_cast<char *>(dst) + off,
	                static_cast<const char *>(src) + off, len);
	}
	return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

template <typename T>
static void
ulm_lam_adddiag(std::ptrdiff_t n, T alpha, T *A, std::ptrdiff_t incRow,
                std::ptrdiff_t incCol)
{
    for (std::ptrdiff_t i = 0; i < n; ++i)
	A[i * (incRow + incCol)] += alpha;
}

template <typename T>
static void
ulm_lam_gecopy(std::ptrdiff_t m, std::ptrdiff_t n, const T *A,
               std::ptrdiff_t incRowA, std::ptrdiff_t incColA, T *B,
               std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    const std::ptrdiff_t bs = 64;
#pragma omp parallel for collapse(2) schedule(static) if (m * n > 1 << 16)
    for (std::ptrdiff_t i0 = 0; i0 < m; i0 += bs)
	for (std::ptrdiff_t j0 = 0; j0 < n; j0 += bs)
	    for (std::ptrdiff_t i = i0; i < i0 + bs && i < m; ++i)
		for (std::ptrdiff_t j = j0; j < j0 + bs && j < n; ++j)
		    B[i * incRowB + j * incColB] = A[i * incRowA + j * incColA];
}

template <typename T>
static void
ulm_lam_sycomplete(bool lower, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
                   std::ptrdiff_t incCol)
{
    for (std::ptrdiff_t i = 0; i < n; ++i)
	for (std::ptrdiff_t j = 0; j < i; ++j) {
	    T &lo = A[i * incRow + j * incCol],
	      &up = A[j * incRow + i * incCol];
	    if (lower)
		up = lo;
	    else
		lo = up;
	}
}
/* blr: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   size_t          in
 *   v4   double[v1][v2]  in    read only
 *   v5   double[v1][1]   in    read only
 *   v6   double[v3][v2]  in    read only
 *   v7   double          in
 *   v8   double          in
 *   v9   double[v3][1]   out   written
 *   v10  double[v3][v3]  out   written
 *   v11  int8_t *        work  workspace (bytes)
 *   v12  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a computed matrix: not positive definite
 *    -12  v12: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
blr(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double *v6,
    double v7, double v8, double *v9, double *v10, int8_t *v11, int64_t *v12)
{
    double v13 = 1.00000000000000000e+00;
    double v14 = 0.0e+00;
    int64_t v15 = 0;
    size_t v16 = -64;
    size_t v17 = 63;
    size_t v18 = 8;
    size_t v19 = 0;
    int64_t v20 = v12[v19];
    size_t v21 = v2 * v18;
    size_t v22 = v21 * v2 + v17 & v16;
    size_t v23 = v22 + (v3 * v18 + v17 & v16);
    size_t v24 = v23 + (v21 + v17 & v16);
    size_t v25 = v24 + (v21 * v3 + v17 & v16);
    ptrdiff_t v26 = (ptrdiff_t)v25;
    int64_t v27 = (int64_t)v26;
    int64_t v28;
    if (v20 == -1) {
	v12[v19] = v27;
	v28 = v15;
    } else {
	int64_t v29;
	if (v20 > v15 & v20 < v27) {
	    v29 = -12;
	} else {
	    bool v30 = v20 == v15;
	    int8_t *v31;
	    if (v30) {
		void *v32 = aligned_alloc(64, v25);
		v31 = (int8_t *)v32;
	    } else {
		v31 = v11;
	    }
	    int8_t *v33 = v31;
	    int64_t v34 = (int64_t)v19;
	    double *v35 = (double *)(v33 + v34);
	    ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v13, v4, 1, v2, v14, v35, v2,
	                                   1);
	    ulmBLAS::gescal<std::ptrdiff_t>(v2, v2, v8, v35, v2, 1);
	    ulm_lam_adddiag(v2, v7, v35, v2, 1);
	    int64_t v36 = ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v35, v2, 1);
	    bool v37 = (bool)(v36 != v15);
	    uint64_t v38 = (uint64_t)v37;
	    int64_t v39 = (int64_t)v38;
	    if (v39 == v15) {
		int64_t v40 = (int64_t)v22;
		double *v41 = (double *)(v33 + v40);
		int64_t v42 = (int64_t)v23;
		double *v43 = (double *)(v33 + v42);
		ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v13, v4, 1, v2, v5, 1,
		                              v14, v43, 1);
		ulmBLAS::trlsv<std::ptrdiff_t>(v2, false, v35, v2, 1, v43, 1);
		ulmBLAS::trusv<std::ptrdiff_t>(v2, false, v35, 1, v2, v43, 1);
		ulmBLAS::gemv<std::ptrdiff_t>(v3, v2, v13, v6, v2, 1, v43, 1,
		                              v14, v41, 1);
		size_t v44 = 8 * v3;
		ulm_lam_copy(v9, v41, v44);
		ulmBLAS::gescal<std::ptrdiff_t>(v3, 1, v8, v9, 1, 1);
		int64_t v45 = (int64_t)v24;
		double *v46 = (double *)(v33 + v45);
		ulm_lam_gecopy(v2, v3, v6, 1, v2, v46, v3, 1);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v2, v3, v35, v2, 1, v46, v3, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(v3, v2, v13, v46, 1, v3, v14,
		                               v10, v3, 1);
		double v47 = v13 / v8;
		ulm_lam_adddiag(v3, v47, v10, v3, 1);
		ulm_lam_sycomplete(true, v3, v10, v3, 1);
	    }
	    if (v30) {
		free(v33);
	    }
	    v29 = v39;
	}
	v28 = v29;
    }
    return v28;
}
