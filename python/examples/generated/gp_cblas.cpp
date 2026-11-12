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
ulm_lam_adddiag(std::ptrdiff_t n, T alpha, T *A, std::ptrdiff_t incRow,
                std::ptrdiff_t incCol)
{
    for (std::ptrdiff_t i = 0; i < n; ++i)
	A[i * (incRow + incCol)] += alpha;
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
 *   v4   double          in
 *   v5   double[v1][v2]  in    read only
 *   v6   double[v2][v2]  in    read only
 *   v7   double[v1][1]   in    read only
 *   v8   double[v2][1]   out   written
 *   v9   double[v2][v2]  out   written
 *   v10  int8_t *        work  workspace (bytes)
 *   v11  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a matrix computed from v3: not positive
 *         definite
 *    -11  v11: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
gp(size_t v1, size_t v2, double *v3, double v4, double *v5, double *v6,
   double *v7, double *v8, double *v9, int8_t *v10, int64_t *v11)
{
    double v12 = -1.00000000000000000e+00;
    double v13 = 1.00000000000000000e+00;
    double v14 = 0.0e+00;
    int64_t v15 = 0;
    size_t v16 = -64;
    size_t v17 = 63;
    size_t v18 = 0;
    int64_t v19 = v11[v18];
    size_t v20 = v1 * 8;
    size_t v21 = v20 * v1 + v17 & v16;
    size_t v22 = v21 + (v20 + v17 & v16);
    size_t v23 = v22 + (v20 * v2 + v17 & v16);
    ptrdiff_t v24 = (ptrdiff_t)v23;
    int64_t v25 = (int64_t)v24;
    int64_t v26;
    if (v19 == -1) {
	v11[v18] = v25;
	v26 = v15;
    } else {
	int64_t v27;
	if (v19 > v15 & v19 < v25) {
	    v27 = -11;
	} else {
	    bool v28 = v19 == v15;
	    int8_t *v29;
	    if (v28) {
		void *v30 = aligned_alloc(64, v23);
		v29 = (int8_t *)v30;
	    } else {
		v29 = v10;
	    }
	    int8_t *v31 = v29;
	    int64_t v32 = (int64_t)v18;
	    double *v33 = (double *)(v31 + v32);
	    size_t v34 = (8 * v1) * v1;
	    ulm_lam_copy(v33, v3, v34);
	    ulm_lam_adddiag(v1, v4, v33, v1, 1);
	    int64_t v35 =
	        ulm_lam_cblas::potrf<std::ptrdiff_t>(true, v1, v33, v1, 1);
	    bool v36 = (bool)(v35 != v15);
	    uint64_t v37 = (uint64_t)v36;
	    int64_t v38 = (int64_t)v37;
	    if (v38 == v15) {
		int64_t v39 = (int64_t)v21;
		double *v40 = (double *)(v31 + v39);
		size_t v41 = 8 * v1;
		ulm_lam_copy(v40, v7, v41);
		ulm_lam_cblas::trlsv<std::ptrdiff_t>(v1, false, v33, v1, 1, v40,
		                                     1);
		ulm_lam_cblas::trusv<std::ptrdiff_t>(v1, false, v33, 1, v1, v40,
		                                     1);
		ulm_lam_cblas::gemv<std::ptrdiff_t>(v2, v1, v13, v5, 1, v2, v40,
		                                    1, v14, v8, 1);
		int64_t v42 = (int64_t)v22;
		double *v43 = (double *)(v31 + v42);
		size_t v44 = (8 * v1) * v2;
		ulm_lam_copy(v43, v5, v44);
		(void)ulm_lam_cblas::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v1, v2, v33, v1, 1, v43, v2, 1);
		size_t v45 = (8 * v2) * v2;
		ulm_lam_copy(v9, v6, v45);
		ulm_lam_cblas::sylrk<std::ptrdiff_t>(v2, v1, v12, v43, 1, v2,
		                                     v13, v9, v2, 1);
		ulm_lam_sycomplete(true, v2, v9, v2, 1);
	    }
	    if (v28) {
		free(v31);
	    }
	    v27 = v38;
	}
	v26 = v27;
    }
    return v26;
}
