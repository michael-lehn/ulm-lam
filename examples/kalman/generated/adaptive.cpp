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
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of a computed matrix: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_adaptive(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 67537600;
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
		void *v16 = aligned_alloc(64, 67537600);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v4;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 = (int64_t)v10;
	    double *v19 = (double *)(v17 + v18);
	    ulm_lam_copy(v19, v2, 32000000);
	    int64_t v20 = (int64_t)32000000;
	    double *v21 = (double *)(v17 + v20);
	    int64_t v22 = (int64_t)64000000;
	    double *v23 = (double *)(v17 + v22);
	    int64_t v24 = (int64_t)64016000;
	    double *v25 = (double *)(v17 + v24);
	    int64_t v26 = (int64_t)64336000;
	    double *v27 = (double *)(v17 + v26);
	    int64_t v28 = (int64_t)67536000;
	    double *v29 = (double *)(v17 + v28);
	    int64_t v30;
	    v30 = v9;
	    for (size_t i31 = v10; i31 < v3; i31 += 1) {
		int64_t v32 = v30;
		int64_t v33;
		if (v32 == v9) {
		    update_covariance(i31, v19);
		    ulm_lam_copy(v21, v19, 32000000);
		    int64_t v34 = ulmBLAS::potrf<std::ptrdiff_t>(true, 2000,
		                                                 v21, 2000, 1);
		    bool v35 = (bool)(v34 != v9);
		    uint64_t v36 = (uint64_t)v35;
		    int64_t v37 = (int64_t)v36;
		    int64_t v38;
		    if (v37 == v9) {
			observe(i31, v23);
			ulm_lam_copy(v27, v1, 3200000);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    true, false, false, false, 2000, 200, v21, 2000, 1,
			    v27, 200, 1);
			ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v7, v27, 1,
			                               200, v8, v25, 200, 1);
			int64_t v39 = ulmBLAS::potrf<std::ptrdiff_t>(
			    true, 200, v25, 200, 1);
			int64_t v40 = v39 != v9 ? 2 : v9;
			if (v40 == v9) {
			    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v21,
			                                   2000, 1, v23, 1);
			    ulmBLAS::gemv<std::ptrdiff_t>(
			        200, 2000, v7, v27, 1, 200, v23, 1, v8, v29, 1);
			    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v25, 200,
			                                   1, v29, 1);
			    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v25, 1,
			                                   200, v29, 1);
			    record(i31, v29);
			}
			v38 = v40;
		    } else {
			v38 = v37;
		    }
		    v33 = v38;
		} else {
		    v33 = v32;
		}
		v30 = v33;
	    }
	    int64_t v41 = v30;
	    if (v14) {
		free(v17);
	    }
	    v13 = v41;
	}
	v12 = v13;
    }
    return v12;
}
