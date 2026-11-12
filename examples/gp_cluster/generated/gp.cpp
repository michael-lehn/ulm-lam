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
/* gp: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v4   double[v1][v2]  in     read only
 *   v5   double[v2][v2]  in     read only
 *   v6   double[v1][1]   in     read only
 *   v7   double[v2][1]   out    written
 *   v8   double[v2][v2]  out    written
 *   v9   int8_t *        work   workspace (bytes)
 *   v10  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of v3: not positive definite
 *    -10  v10: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
gp(size_t v1, size_t v2, double *v3, double *v4, double *v5, double *v6,
   double *v7, double *v8, int8_t *v9, int64_t *v10)
{
    double v11 = -1.00000000000000000e+00;
    double v12 = 1.00000000000000000e+00;
    double v13 = 0.0e+00;
    int64_t v14 = 0;
    size_t v15 = -64;
    size_t v16 = 63;
    size_t v17 = 0;
    int64_t v18 = v10[v17];
    size_t v19 = v1 * 8;
    size_t v20 = v19 + v16 & v15;
    size_t v21 = v20 + (v19 * v2 + v16 & v15);
    ptrdiff_t v22 = (ptrdiff_t)v21;
    int64_t v23 = (int64_t)v22;
    int64_t v24;
    if (v18 == -1) {
	v10[v17] = v23;
	v24 = v14;
    } else {
	int64_t v25;
	if (v18 > v14 & v18 < v23) {
	    v25 = -10;
	} else {
	    bool v26 = v18 == v14;
	    int8_t *v27;
	    if (v26) {
		void *v28 = aligned_alloc(64, v21);
		v27 = (int8_t *)v28;
	    } else {
		v27 = v9;
	    }
	    int8_t *v29 = v27;
	    int64_t v30 = ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v3, v1, 1);
	    bool v31 = (bool)(v30 != v14);
	    uint64_t v32 = (uint64_t)v31;
	    int64_t v33 = (int64_t)v32;
	    if (v33 == v14) {
		int64_t v34 = (int64_t)v17;
		double *v35 = (double *)(v29 + v34);
		size_t v36 = 8 * v1;
		ulm_lam_copy(v35, v6, v36);
		ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v3, v1, 1, v35, 1);
		ulmBLAS::trusv<std::ptrdiff_t>(v1, false, v3, 1, v1, v35, 1);
		ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v12, v4, 1, v2, v35, 1,
		                              v13, v7, 1);
		int64_t v37 = (int64_t)v20;
		double *v38 = (double *)(v29 + v37);
		size_t v39 = (8 * v1) * v2;
		ulm_lam_copy(v38, v4, v39);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v1, v2, v3, v1, 1, v38, v2, 1);
		size_t v40 = (8 * v2) * v2;
		ulm_lam_copy(v8, v5, v40);
		ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v11, v38, 1, v2, v12, v8,
		                               v2, 1);
		ulm_lam_sycomplete(true, v2, v8, v2, 1);
	    }
	    if (v26) {
		free(v29);
	    }
	    v25 = v33;
	}
	v24 = v25;
    }
    return v24;
}
