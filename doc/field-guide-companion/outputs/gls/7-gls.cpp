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

/* gls: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  double[v2][v1]  in    read only
 *   v4  double[v1][v1]  in    read only
 *   v5  double[v1][1]   in    read only
 *   v6  double[v2][1]   out   written
 *   v7  int8_t *        work  workspace (bytes)
 *   v8  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v4: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -8  v8: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
gls(size_t v1, size_t v2, double *v3, double *v4, double *v5, double *v6,
    int8_t *v7, int64_t *v8)
{
    double v9 = 1.00000000000000000e+00;
    double v10 = 0.0e+00;
    int64_t v11 = 0;
    size_t v12 = -64;
    size_t v13 = 63;
    size_t v14 = 8;
    size_t v15 = 0;
    int64_t v16 = v8[v15];
    size_t v17 = v1 * v14;
    size_t v18 = v17 * v1 + v13 & v12;
    size_t v19 = v18 + ((v2 * v14) * v2 + v13 & v12);
    size_t v20 = v19 + (v17 * v2 + v13 & v12);
    size_t v21 = v20 + (v17 + v13 & v12);
    ptrdiff_t v22 = (ptrdiff_t)v21;
    int64_t v23 = (int64_t)v22;
    int64_t v24;
    if (v16 == -1) {
	v8[v15] = v23;
	v24 = v11;
    } else {
	int64_t v25;
	if (v16 > v11 & v16 < v23) {
	    v25 = -8;
	} else {
	    bool v26 = v16 == v11;
	    int8_t *v27;
	    if (v26) {
		void *v28 = aligned_alloc(64, v21);
		v27 = (int8_t *)v28;
	    } else {
		v27 = v7;
	    }
	    int8_t *v29 = v27;
	    int64_t v30 = (int64_t)v15;
	    double *v31 = (double *)(v29 + v30);
	    size_t v32 = (8 * v1) * v1;
	    ulm_lam_copy(v31, v4, v32);
	    int64_t v33 =
	        ulm_lam_cblas::potrf<std::ptrdiff_t>(true, v1, v31, v1, 1);
	    bool v34 = (bool)(v33 != v11);
	    uint64_t v35 = (uint64_t)v34;
	    int64_t v36 = (int64_t)v35;
	    int64_t v37;
	    if (v36 == v11) {
		int64_t v38 = (int64_t)v18;
		double *v39 = (double *)(v29 + v38);
		int64_t v40 = (int64_t)v19;
		double *v41 = (double *)(v29 + v40);
		ulm_lam_gecopy(v1, v2, v3, 1, v1, v41, v2, 1);
		(void)ulm_lam_cblas::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v1, v2, v31, v1, 1, v41, v2, 1);
		ulm_lam_cblas::sylrk<std::ptrdiff_t>(v2, v1, v9, v41, 1, v2,
		                                     v10, v39, v2, 1);
		int64_t v42 =
		    ulm_lam_cblas::potrf<std::ptrdiff_t>(true, v2, v39, v2, 1);
		int64_t v43 = v42 != v11 ? 2 : v11;
		if (v43 == v11) {
		    int64_t v44 = (int64_t)v20;
		    double *v45 = (double *)(v29 + v44);
		    size_t v46 = 8 * v1;
		    ulm_lam_copy(v45, v5, v46);
		    ulm_lam_cblas::trlsv<std::ptrdiff_t>(v1, false, v31, v1, 1,
		                                         v45, 1);
		    ulm_lam_cblas::gemv<std::ptrdiff_t>(v2, v1, v9, v41, 1, v2,
		                                        v45, 1, v10, v6, 1);
		    ulm_lam_cblas::trlsv<std::ptrdiff_t>(v2, false, v39, v2, 1,
		                                         v6, 1);
		    ulm_lam_cblas::trusv<std::ptrdiff_t>(v2, false, v39, 1, v2,
		                                         v6, 1);
		}
		v37 = v43;
	    } else {
		v37 = v36;
	    }
	    int64_t v47 = v37;
	    if (v26) {
		free(v29);
	    }
	    v25 = v47;
	}
	v24 = v25;
    }
    return v24;
}
