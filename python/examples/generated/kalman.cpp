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

extern "C" void
observe(size_t, double *);
extern "C" void
record(size_t, double *);
/* estimate: matrices by reference (row-major), owned by the caller
 *   v1  double[4000][200]   inout  may be overwritten (unspecified on exit)
 *   v2  double[4000][4000]  inout  may be overwritten (unspecified on exit)
 *   v3  size_t              in
 *   v4  int8_t *            work   workspace (bytes)
 *   v5  int64_t *           info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 6753600;
    double v7 = 1.00000000000000000e+00;
    double v8 = 0.0e+00;
    int64_t v9 = 0;
    size_t v10 = 0;
    int64_t v11 = v5[v10];
    int64_t v12;
    if (v11 == -1) {
	v5[v10] = v6;
	v12 = v9;
    } else {
	int64_t v13;
	if (v11 > v9 & v11 < v6) {
	    v13 = -5;
	} else {
	    bool v14 = v11 == v9;
	    int8_t *v15;
	    if (v14) {
		void *v16 = aligned_alloc(64, 6753600);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v4;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 =
	        ulmBLAS::potrf<std::ptrdiff_t>(true, 4000, v2, 4000, 1);
	    bool v19 = (bool)(v18 != v9);
	    uint64_t v20 = (uint64_t)v19;
	    int64_t v21 = (int64_t)v20;
	    int64_t v22;
	    if (v21 == v9) {
		int64_t v23 = (int64_t)v10;
		double *v24 = (double *)(v17 + v23);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false,
		                                     4000, 200, v2, 4000, 1, v1,
		                                     200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 4000, v7, v1, 1, 200, v8,
		                               v24, 200, 1);
		int64_t v25 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v24, 200, 1);
		int64_t v26 = v25 != v9 ? 2 : v9;
		if (v26 == v9) {
		    ptrdiff_t v27 = (ptrdiff_t)v3;
		    ptrdiff_t v28 = (ptrdiff_t)v10;
		    bool v29 = (v27 > v28 ? v3 : v10) * 16000000 + 3520800000 <
		               (v27 > v28 ? v3 : v10) * 176800000;
		    int64_t v30 = (int64_t)320000;
		    double *v31 = (double *)(v17 + v30);
		    if (v29) {
			ulm_lam_gecopy(200, 4000, v1, 1, 200, v31, 4000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    true, false, false, false, 200, 4000, v24, 200, 1,
			    v31, 4000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, 200, 4000, v24, 1, 200,
			    v31, 4000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, 4000, 200, v2, 1, 4000,
			    v31, 1, 4000);
		    }
		    int64_t v32 = (int64_t)6720000;
		    double *v33 = (double *)(v17 + v32);
		    int64_t v34 = (int64_t)6752000;
		    double *v35 = (double *)(v17 + v34);
		    for (size_t i36 = v10; i36 < v3; i36 += 1) {
			observe(i36, v33);
			if (v29) {
			    ulmBLAS::gemv<std::ptrdiff_t>(200, 4000, v7, v31,
			                                  4000, 1, v33, 1, v8,
			                                  v35, 1);
			} else {
			    ulmBLAS::trlsv<std::ptrdiff_t>(4000, false, v2,
			                                   4000, 1, v33, 1);
			    ulmBLAS::gemv<std::ptrdiff_t>(
			        200, 4000, v7, v1, 1, 200, v33, 1, v8, v35, 1);
			    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v24, 200,
			                                   1, v35, 1);
			    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v24, 1,
			                                   200, v35, 1);
			}
			record(i36, v35);
		    }
		}
		v22 = v26;
	    } else {
		v22 = v21;
	    }
	    int64_t v37 = v22;
	    if (v14) {
		free(v17);
	    }
	    v13 = v37;
	}
	v12 = v13;
    }
    return v12;
}
/* estimate_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[4000][200]   in    read only
 *   v2  double[4000][4000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_keep(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    int64_t v7 = -1;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10 = (int64_t)v8;
    double *v11 = (double *)(v4 + v10);
    int64_t v12 = (int64_t)1;
    double *v13 = (double *)(v4 + v12);
    v5[v8] = v7;
    int64_t v14 = estimate(v11, v13, v3, v4, v5);
    int64_t v15;
    if (v14 == v6) {
	int64_t v16 = v5[v8];
	uint64_t v17 = (uint64_t)v16;
	uint64_t v18 = (uint64_t)63;
	int64_t v19 = (int64_t)(v17 + v18);
	uint64_t v20 = (uint64_t)v19;
	uint64_t v21 = (uint64_t)-64;
	int64_t v22 = (int64_t)(v20 & v21);
	uint64_t v23 = (uint64_t)v22;
	uint64_t v24 = (uint64_t)134400000;
	int64_t v25 = (int64_t)(v23 + v24);
	v5[v8] = v9;
	int64_t v26;
	if (v9 == v7) {
	    v5[v8] = v25;
	    v26 = v6;
	} else {
	    int64_t v27;
	    if (v9 > v6 & v9 < v25) {
		v27 = -5;
	    } else {
		bool v28 = v9 == v6;
		int8_t *v29;
		if (v28) {
		    ptrdiff_t v30 = (ptrdiff_t)v25;
		    size_t v31 = (size_t)v30;
		    void *v32 = aligned_alloc(64, v31);
		    v29 = (int8_t *)v32;
		} else {
		    v29 = v4;
		}
		int8_t *v33 = v29;
		int64_t v34 = (int64_t)v8;
		double *v35 = (double *)(v33 + v34);
		ulm_lam_copy(v35, v1, 6400000);
		int64_t v36 = (int64_t)6400000;
		double *v37 = (double *)(v33 + v36);
		ulm_lam_copy(v37, v2, 128000000);
		int64_t v38 = (int64_t)134400000;
		v5[v8] = v16;
		int64_t v39 = estimate(v35, v37, v3, v33 + v38, v5);
		if (v39 == v6) {
		    v5[v8] = v9;
		}
		if (v28) {
		    free(v33);
		}
		v27 = v39;
	    }
	    v26 = v27;
	}
	v15 = v26;
    } else {
	v15 = v14;
    }
    return v15;
}
