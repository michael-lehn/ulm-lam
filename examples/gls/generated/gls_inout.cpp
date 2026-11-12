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

/* gls_inout: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in     read only
 *   v2  double[2000][2000]  inout  may be overwritten (unspecified on exit)
 *   v3  double[2000][1]     inout  may be overwritten (unspecified on exit)
 *   v4  double[200][1]      out    written
 *   v5  int8_t *            work   workspace (bytes)
 *   v6  int64_t *           info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
gls_inout(double *v1, double *v2, double *v3, double *v4, int8_t *v5,
          int64_t *v6)
{
    int64_t v7 = 3520000;
    double v8 = 1.00000000000000000e+00;
    double v9 = 0.0e+00;
    int64_t v10 = 0;
    size_t v11 = 0;
    int64_t v12 = v6[v11];
    int64_t v13;
    if (v12 == -1) {
	v6[v11] = v7;
	v13 = v10;
    } else {
	int64_t v14;
	if (v12 > v10 & v12 < v7) {
	    v14 = -6;
	} else {
	    bool v15 = v12 == v10;
	    int8_t *v16;
	    if (v15) {
		void *v17 = aligned_alloc(64, 3520000);
		v16 = (int8_t *)v17;
	    } else {
		v16 = v5;
	    }
	    int8_t *v18 = v16;
	    int64_t v19 =
	        ulmBLAS::potrf<std::ptrdiff_t>(true, 2000, v2, 2000, 1);
	    bool v20 = (bool)(v19 != v10);
	    uint64_t v21 = (uint64_t)v20;
	    int64_t v22 = (int64_t)v21;
	    int64_t v23;
	    if (v22 == v10) {
		int64_t v24 = (int64_t)v11;
		double *v25 = (double *)(v18 + v24);
		int64_t v26 = (int64_t)320000;
		double *v27 = (double *)(v18 + v26);
		ulm_lam_copy(v27, v1, 3200000);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false,
		                                     2000, 200, v2, 2000, 1,
		                                     v27, 200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v8, v27, 1, 200, v9,
		                               v25, 200, 1);
		int64_t v28 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v25, 200, 1);
		int64_t v29 = v28 != v10 ? 2 : v10;
		if (v29 == v10) {
		    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v2, 2000, 1, v3,
		                                   1);
		    ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v8, v27, 1, 200,
		                                  v3, 1, v9, v4, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v25, 200, 1, v4,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v25, 1, 200, v4,
		                                   1);
		}
		v23 = v29;
	    } else {
		v23 = v22;
	    }
	    int64_t v30 = v23;
	    if (v15) {
		free(v18);
	    }
	    v14 = v30;
	}
	v13 = v14;
    }
    return v13;
}
