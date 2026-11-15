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

/* gls: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  double[v1][v2]  inout  may be overwritten (unspecified on exit)
 *   v4  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v5  double[v1][1]   inout  may be overwritten (unspecified on exit)
 *   v6  double[v2][1]   out    written
 *   v7  int8_t *        work   workspace (bytes)
 *   v8  int64_t *       info   -1: query, 0: allocate, >0: size of work
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
    size_t v12 = 0;
    int64_t v13 = v8[v12];
    size_t v14 = (v2 * 8) * v2 + 63 & -64;
    ptrdiff_t v15 = (ptrdiff_t)v14;
    int64_t v16 = (int64_t)v15;
    int64_t v17;
    if (v13 == -1) {
	v8[v12] = v16;
	v17 = v11;
    } else {
	int64_t v18;
	if (v13 > v11 & v13 < v16) {
	    v18 = -8;
	} else {
	    bool v19 = v13 == v11;
	    int8_t *v20;
	    if (v19) {
		void *v21 = aligned_alloc(64, v14);
		v20 = (int8_t *)v21;
	    } else {
		v20 = v7;
	    }
	    int8_t *v22 = v20;
	    int64_t v23 = ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v4, v1, 1);
	    bool v24 = (bool)(v23 != v11);
	    uint64_t v25 = (uint64_t)v24;
	    int64_t v26 = (int64_t)v25;
	    int64_t v27;
	    if (v26 == v11) {
		int64_t v28 = (int64_t)v12;
		double *v29 = (double *)(v22 + v28);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(
		    true, false, false, false, v1, v2, v4, v1, 1, v3, v2, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v9, v3, 1, v2, v10, v29,
		                               v2, 1);
		int64_t v30 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v29, v2, 1);
		int64_t v31 = v30 != v11 ? 2 : v11;
		if (v31 == v11) {
		    ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v4, v1, 1, v5, 1);
		    ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v9, v3, 1, v2, v5, 1,
		                                  v10, v6, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(v2, false, v29, v2, 1, v6,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(v2, false, v29, 1, v2, v6,
		                                   1);
		}
		v27 = v31;
	    } else {
		v27 = v26;
	    }
	    int64_t v32 = v27;
	    if (v19) {
		free(v22);
	    }
	    v18 = v32;
	}
	v17 = v18;
    }
    return v17;
}
/* gls_keep: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  double[v1][v2]  in    read only
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
gls_keep(size_t v1, size_t v2, double *v3, double *v4, double *v5, double *v6,
         int8_t *v7, int64_t *v8)
{
    int64_t v9 = 0;
    int64_t v10 = -1;
    size_t v11 = -64;
    size_t v12 = 63;
    size_t v13 = 0;
    int64_t v14 = v8[v13];
    size_t v15 = v1 * 8;
    size_t v16 = v15 * v2 + v12 & v11;
    size_t v17 = v16 + (v15 * v1 + v12 & v11);
    size_t v18 = v17 + (v15 + v12 & v11);
    ptrdiff_t v19 = (ptrdiff_t)v18;
    int64_t v20 = (int64_t)v19;
    int64_t v21 = (int64_t)2;
    double *v22 = (double *)(v7 + v21);
    int64_t v23 = (int64_t)3;
    double *v24 = (double *)(v7 + v23);
    int64_t v25 = (int64_t)4;
    double *v26 = (double *)(v7 + v25);
    int64_t v27 = (int64_t)5;
    double *v28 = (double *)(v7 + v27);
    v8[v13] = v10;
    int64_t v29 = gls(v1, v2, v22, v24, v26, v28, v7, v8);
    int64_t v30;
    if (v29 == v9) {
	int64_t v31 = v8[v13];
	uint64_t v32 = (uint64_t)v31;
	uint64_t v33 = (uint64_t)63;
	int64_t v34 = (int64_t)(v32 + v33);
	uint64_t v35 = (uint64_t)v34;
	uint64_t v36 = (uint64_t)-64;
	int64_t v37 = (int64_t)(v35 & v36);
	uint64_t v38 = (uint64_t)v20;
	uint64_t v39 = (uint64_t)v37;
	int64_t v40 = (int64_t)(v38 + v39);
	v8[v13] = v14;
	int64_t v41;
	if (v14 == v10) {
	    v8[v13] = v40;
	    v41 = v9;
	} else {
	    int64_t v42;
	    if (v14 > v9 & v14 < v40) {
		v42 = -8;
	    } else {
		bool v43 = v14 == v9;
		int8_t *v44;
		if (v43) {
		    ptrdiff_t v45 = (ptrdiff_t)v40;
		    size_t v46 = (size_t)v45;
		    void *v47 = aligned_alloc(64, v46);
		    v44 = (int8_t *)v47;
		} else {
		    v44 = v7;
		}
		int8_t *v48 = v44;
		int64_t v49 = (int64_t)v13;
		double *v50 = (double *)(v48 + v49);
		size_t v51 = (8 * v1) * v2;
		ulm_lam_copy(v50, v3, v51);
		int64_t v52 = (int64_t)v16;
		double *v53 = (double *)(v48 + v52);
		size_t v54 = (8 * v1) * v1;
		ulm_lam_copy(v53, v4, v54);
		int64_t v55 = (int64_t)v17;
		double *v56 = (double *)(v48 + v55);
		size_t v57 = 8 * v1;
		ulm_lam_copy(v56, v5, v57);
		int64_t v58 = (int64_t)v18;
		v8[v13] = v31;
		int64_t v59 = gls(v1, v2, v50, v53, v56, v6, v48 + v58, v8);
		if (v59 == v9) {
		    v8[v13] = v14;
		}
		if (v43) {
		    free(v48);
		}
		v42 = v59;
	    }
	    v41 = v42;
	}
	v30 = v41;
    } else {
	v30 = v29;
    }
    return v30;
}
