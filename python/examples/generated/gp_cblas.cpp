#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulm_lam/cblas_backend.h>
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
 *   v3   double[v1][v1]  in    read only
 *   v4   double[v1][v2]  in    read only
 *   v5   double[v2][v2]  in    read only
 *   v6   double[v1][1]   in    read only
 *   v7   double[v2][1]   out   written
 *   v8   double[v2][v2]  out   written
 *   v9   int8_t *        work  workspace (bytes)
 *   v10  int64_t *       info  -1: query, 0: allocate, >0: size of work
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
    size_t v20 = v19 * v1 + v16 & v15;
    size_t v21 = v20 + (v19 + v16 & v15);
    size_t v22 = v21 + (v19 * v2 + v16 & v15);
    ptrdiff_t v23 = (ptrdiff_t)v22;
    int64_t v24 = (int64_t)v23;
    int64_t v25;
    if (v18 == -1) {
	v10[v17] = v24;
	v25 = v14;
    } else {
	int64_t v26;
	if (v18 > v14 & v18 < v24) {
	    v26 = -10;
	} else {
	    bool v27 = v18 == v14;
	    int8_t *v28;
	    if (v27) {
		void *v29 = aligned_alloc(64, v22);
		v28 = (int8_t *)v29;
	    } else {
		v28 = v9;
	    }
	    int8_t *v30 = v28;
	    int64_t v31 = (int64_t)v17;
	    double *v32 = (double *)(v30 + v31);
	    size_t v33 = (8 * v1) * v1;
	    ulm_lam_copy(v32, v3, v33);
	    int64_t v34 =
	        ulm_lam_cblas::potrf<std::ptrdiff_t>(true, v1, v32, v1, 1);
	    bool v35 = (bool)(v34 != v14);
	    uint64_t v36 = (uint64_t)v35;
	    int64_t v37 = (int64_t)v36;
	    if (v37 == v14) {
		int64_t v38 = (int64_t)v20;
		double *v39 = (double *)(v30 + v38);
		size_t v40 = 8 * v1;
		ulm_lam_copy(v39, v6, v40);
		ulm_lam_cblas::trlsv<std::ptrdiff_t>(v1, false, v32, v1, 1, v39,
		                                     1);
		ulm_lam_cblas::trusv<std::ptrdiff_t>(v1, false, v32, 1, v1, v39,
		                                     1);
		ulm_lam_cblas::gemv<std::ptrdiff_t>(v2, v1, v12, v4, 1, v2, v39,
		                                    1, v13, v7, 1);
		int64_t v41 = (int64_t)v21;
		double *v42 = (double *)(v30 + v41);
		size_t v43 = (8 * v1) * v2;
		ulm_lam_copy(v42, v4, v43);
		(void)ulm_lam_cblas::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v1, v2, v32, v1, 1, v42, v2, 1);
		size_t v44 = (8 * v2) * v2;
		ulm_lam_copy(v8, v5, v44);
		ulm_lam_cblas::sylrk<std::ptrdiff_t>(v2, v1, v11, v42, 1, v2,
		                                     v12, v8, v2, 1);
		ulm_lam_sycomplete(true, v2, v8, v2, 1);
	    }
	    if (v27) {
		free(v30);
	    }
	    v26 = v37;
	}
	v25 = v26;
    }
    return v25;
}
