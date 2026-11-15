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

extern "C" void
update_covariance(size_t, double *);
extern "C" void
observe(size_t, double *);
extern "C" void
record(size_t, double *);
/* estimate_adaptive: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   inout  may be overwritten (unspecified on exit)
 *   v2  double[2000][2000]  inout  may be overwritten (unspecified on exit)
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
estimate_adaptive(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 35537600;
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
		void *v16 = aligned_alloc(64, 35537600);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v4;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 = (int64_t)v10;
	    double *v19 = (double *)(v17 + v18);
	    int64_t v20 = (int64_t)32000000;
	    double *v21 = (double *)(v17 + v20);
	    int64_t v22 = (int64_t)32016000;
	    double *v23 = (double *)(v17 + v22);
	    int64_t v24 = (int64_t)32336000;
	    double *v25 = (double *)(v17 + v24);
	    int64_t v26 = (int64_t)35536000;
	    double *v27 = (double *)(v17 + v26);
	    int64_t v28;
	    v28 = v9;
	    for (size_t i29 = v10; i29 < v3; i29 += 1) {
		int64_t v30 = v28;
		int64_t v31;
		if (v30 == v9) {
		    update_covariance(i29, v2);
		    ulm_lam_copy(v19, v2, 32000000);
		    int64_t v32 = ulmBLAS::potrf<std::ptrdiff_t>(true, 2000,
		                                                 v19, 2000, 1);
		    bool v33 = (bool)(v32 != v9);
		    uint64_t v34 = (uint64_t)v33;
		    int64_t v35 = (int64_t)v34;
		    int64_t v36;
		    if (v35 == v9) {
			observe(i29, v21);
			ulm_lam_copy(v25, v1, 3200000);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    true, false, false, false, 2000, 200, v19, 2000, 1,
			    v25, 200, 1);
			ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v7, v25, 1,
			                               200, v8, v23, 200, 1);
			int64_t v37 = ulmBLAS::potrf<std::ptrdiff_t>(
			    true, 200, v23, 200, 1);
			int64_t v38 = v37 != v9 ? 2 : v9;
			if (v38 == v9) {
			    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v19,
			                                   2000, 1, v21, 1);
			    ulmBLAS::gemv<std::ptrdiff_t>(
			        200, 2000, v7, v25, 1, 200, v21, 1, v8, v27, 1);
			    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v23, 200,
			                                   1, v27, 1);
			    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v23, 1,
			                                   200, v27, 1);
			    record(i29, v27);
			}
			v36 = v38;
		    } else {
			v36 = v35;
		    }
		    v31 = v36;
		} else {
		    v31 = v30;
		}
		v28 = v31;
	    }
	    int64_t v39 = v28;
	    if (v14) {
		free(v17);
	    }
	    v13 = v39;
	}
	v12 = v13;
    }
    return v12;
}
/* estimate_adaptive_keep: matrices by reference (row-major), owned by the
 * caller v1  double[2000][200]   in    read only v2  double[2000][2000]  in
 * read only v3  size_t              in v4  int8_t *            work  workspace
 * (bytes) v5  int64_t *           info  -1: query, 0: allocate, >0: size of
 * work returns info: 0  success 1  Cholesky factorisation of v2: not positive
 * definite 2  Cholesky factorisation of a computed matrix: not positive
 * definite -5  v5: workspace too small (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_adaptive_keep(double *v1, double *v2, size_t v3, int8_t *v4,
                       int64_t *v5)
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
    int64_t v14 = estimate_adaptive(v11, v13, v3, v4, v5);
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
	uint64_t v24 = (uint64_t)32000000;
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
		ulm_lam_copy(v35, v2, 32000000);
		int64_t v36 = (int64_t)32000000;
		v5[v8] = v16;
		int64_t v37 = estimate_adaptive(v1, v35, v3, v33 + v36, v5);
		if (v37 == v6) {
		    v5[v8] = v9;
		}
		if (v28) {
		    free(v33);
		}
		v27 = v37;
	    }
	    v26 = v27;
	}
	v15 = v26;
    } else {
	v15 = v14;
    }
    return v15;
}
