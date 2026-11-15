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
observe(size_t, double *);
extern "C" void
postprocess(size_t, double *, int8_t *, int64_t *);
/* gls: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   inout  may be overwritten (unspecified on exit)
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
gls(double *v1, double *v2, double *v3, double *v4, int8_t *v5, int64_t *v6)
{
    int64_t v7 = 320000;
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
		void *v17 = aligned_alloc(64, 320000);
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
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false,
		                                     2000, 200, v2, 2000, 1, v1,
		                                     200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v8, v1, 1, 200, v9,
		                               v25, 200, 1);
		int64_t v26 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v25, 200, 1);
		int64_t v27 = v26 != v10 ? 2 : v10;
		if (v27 == v10) {
		    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v2, 2000, 1, v3,
		                                   1);
		    ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v8, v1, 1, 200, v3,
		                                  1, v9, v4, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v25, 200, 1, v4,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v25, 1, 200, v4,
		                                   1);
		}
		v23 = v27;
	    } else {
		v23 = v22;
	    }
	    int64_t v28 = v23;
	    if (v15) {
		free(v18);
	    }
	    v14 = v28;
	}
	v13 = v14;
    }
    return v13;
}
/* gls_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  double[2000][1]     in    read only
 *   v4  double[200][1]      out   written
 *   v5  int8_t *            work  workspace (bytes)
 *   v6  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
gls_keep(double *v1, double *v2, double *v3, double *v4, int8_t *v5,
         int64_t *v6)
{
    int64_t v7 = 0;
    int64_t v8 = -1;
    size_t v9 = 0;
    int64_t v10 = v6[v9];
    int64_t v11 = (int64_t)v9;
    double *v12 = (double *)(v5 + v11);
    int64_t v13 = (int64_t)1;
    double *v14 = (double *)(v5 + v13);
    int64_t v15 = (int64_t)2;
    double *v16 = (double *)(v5 + v15);
    int64_t v17 = (int64_t)3;
    double *v18 = (double *)(v5 + v17);
    v6[v9] = v8;
    int64_t v19 = gls(v12, v14, v16, v18, v5, v6);
    int64_t v20;
    if (v19 == v7) {
	int64_t v21 = v6[v9];
	uint64_t v22 = (uint64_t)v21;
	uint64_t v23 = (uint64_t)63;
	int64_t v24 = (int64_t)(v22 + v23);
	uint64_t v25 = (uint64_t)v24;
	uint64_t v26 = (uint64_t)-64;
	int64_t v27 = (int64_t)(v25 & v26);
	uint64_t v28 = (uint64_t)v27;
	uint64_t v29 = (uint64_t)35216000;
	int64_t v30 = (int64_t)(v28 + v29);
	v6[v9] = v10;
	int64_t v31;
	if (v10 == v8) {
	    v6[v9] = v30;
	    v31 = v7;
	} else {
	    int64_t v32;
	    if (v10 > v7 & v10 < v30) {
		v32 = -6;
	    } else {
		bool v33 = v10 == v7;
		int8_t *v34;
		if (v33) {
		    ptrdiff_t v35 = (ptrdiff_t)v30;
		    size_t v36 = (size_t)v35;
		    void *v37 = aligned_alloc(64, v36);
		    v34 = (int8_t *)v37;
		} else {
		    v34 = v5;
		}
		int8_t *v38 = v34;
		int64_t v39 = (int64_t)v9;
		double *v40 = (double *)(v38 + v39);
		ulm_lam_copy(v40, v1, 3200000);
		int64_t v41 = (int64_t)3200000;
		double *v42 = (double *)(v38 + v41);
		ulm_lam_copy(v42, v2, 32000000);
		int64_t v43 = (int64_t)35200000;
		double *v44 = (double *)(v38 + v43);
		ulm_lam_copy(v44, v3, 16000);
		int64_t v45 = (int64_t)35216000;
		v6[v9] = v21;
		int64_t v46 = gls(v40, v42, v44, v4, v38 + v45, v6);
		if (v46 == v7) {
		    v6[v9] = v10;
		}
		if (v33) {
		    free(v38);
		}
		v32 = v46;
	    }
	    v31 = v32;
	}
	v20 = v31;
    } else {
	v20 = v19;
    }
    return v20;
}
/* estimate_all: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   inout  may be overwritten (unspecified on exit)
 *   v2  double[2000][2000]  inout  may be overwritten (unspecified on exit)
 *   v3  size_t              in
 *   v4  int8_t *            work   workspace (bytes)
 *   v5  int64_t *           info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of gls returned info != 0
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_all(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    int64_t v7 = -64;
    int64_t v8 = 63;
    int64_t v9 = -1;
    size_t v10 = 1;
    size_t v11 = 0;
    int64_t v12 = v5[v11];
    int64_t v13 = (int64_t)v11;
    double *v14 = (double *)(v4 + v13);
    int64_t v15 = (int64_t)v10;
    double *v16 = (double *)(v4 + v15);
    int64_t v17 = (int64_t)2;
    double *v18 = (double *)(v4 + v17);
    int64_t v19 = (int64_t)3;
    double *v20 = (double *)(v4 + v19);
    v5[v11] = v9;
    int64_t v21 = gls(v14, v16, v18, v20, v4, v5);
    int64_t v22;
    if (v21 == v6) {
	int64_t v23 = v5[v11];
	uint64_t v24 = (uint64_t)v23;
	uint64_t v25 = (uint64_t)v8;
	int64_t v26 = (int64_t)(v24 + v25);
	uint64_t v27 = (uint64_t)v26;
	uint64_t v28 = (uint64_t)v7;
	int64_t v29 = (int64_t)(v27 & v28);
	uint64_t v30 = (uint64_t)v29;
	uint64_t v31 = (uint64_t)35217600;
	int64_t v32 = (int64_t)(v30 + v31);
	int64_t v33 = (int64_t)v10;
	double *v34 = (double *)(v4 + v33);
	v5[v11] = v9;
	postprocess(v11, v34, v4, v5);
	int64_t v35 = v5[v11];
	uint64_t v36 = (uint64_t)v35;
	uint64_t v37 = (uint64_t)v8;
	int64_t v38 = (int64_t)(v36 + v37);
	uint64_t v39 = (uint64_t)v38;
	uint64_t v40 = (uint64_t)v7;
	int64_t v41 = (int64_t)(v39 & v40);
	uint64_t v42 = (uint64_t)v32;
	uint64_t v43 = (uint64_t)v41;
	int64_t v44 = (int64_t)(v42 + v43);
	v5[v11] = v12;
	int64_t v45;
	if (v12 == v9) {
	    v5[v11] = v44;
	    v45 = v6;
	} else {
	    int64_t v46;
	    if (v12 > v6 & v12 < v44) {
		v46 = -5;
	    } else {
		bool v47 = v12 == v6;
		int8_t *v48;
		if (v47) {
		    ptrdiff_t v49 = (ptrdiff_t)v44;
		    size_t v50 = (size_t)v49;
		    void *v51 = aligned_alloc(64, v50);
		    v48 = (int8_t *)v51;
		} else {
		    v48 = v4;
		}
		int8_t *v52 = v48;
		int64_t v53 = (int64_t)v11;
		double *v54 = (double *)(v52 + v53);
		int64_t v55 = (int64_t)16000;
		double *v56 = (double *)(v52 + v55);
		int64_t v57 = (int64_t)17600;
		double *v58 = (double *)(v52 + v57);
		int64_t v59 = (int64_t)3217600;
		double *v60 = (double *)(v52 + v59);
		int64_t v61;
		v61 = v6;
		for (size_t i62 = v11; i62 < v3; i62 += v10) {
		    int64_t v63 = v61;
		    int64_t v64;
		    if (v63 == v6) {
			observe(i62, v54);
			ulm_lam_copy(v58, v1, 3200000);
			ulm_lam_copy(v60, v2, 32000000);
			int64_t v65 = (int64_t)35217600;
			v5[v11] = v23;
			bool v66 = (bool)(gls(v58, v60, v54, v56, v52 + v65,
			                      v5) != v6);
			uint64_t v67 = (uint64_t)v66;
			int64_t v68 = (int64_t)v67;
			if (v68 == v6) {
			    ptrdiff_t v69 = (ptrdiff_t)v32;
			    size_t v70 = (size_t)v69;
			    int64_t v71 = (int64_t)v70;
			    int8_t *v72 = v52 + v71;
			    v5[v11] = v35;
			    postprocess(i62, v56, v72, v5);
			}
			v64 = v68;
		    } else {
			v64 = v63;
		    }
		    v61 = v64;
		}
		int64_t v73 = v61;
		if (v73 == v6) {
		    v5[v11] = v12;
		}
		if (v47) {
		    free(v52);
		}
		v46 = v73;
	    }
	    v45 = v46;
	}
	v22 = v45;
    } else {
	v22 = v21;
    }
    return v22;
}
/* estimate_all_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of gls returned info != 0
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_all_keep(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
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
    int64_t v14 = estimate_all(v11, v13, v3, v4, v5);
    int64_t v15;
    if (v14 == v6) {
	int64_t v16 = v5[v8];
	uint64_t v17 = (uint64_t)v16;
	uint64_t v18 = (uint64_t)63;
	int64_t v19 = (int64_t)(v17 + v18);
	uint64_t v20 = (uint64_t)v19;
	uint64_t v21 = (uint64_t)-64;
	int64_t v22 = (int64_t)(v20 & v21);
	v5[v8] = v9;
	int64_t v23;
	if (v9 == v7) {
	    v5[v8] = v22;
	    v23 = v6;
	} else {
	    int64_t v24;
	    if (v9 > v6 & v9 < v22) {
		v24 = -5;
	    } else {
		bool v25 = v9 == v6;
		int8_t *v26;
		if (v25) {
		    ptrdiff_t v27 = (ptrdiff_t)v22;
		    size_t v28 = (size_t)v27;
		    void *v29 = aligned_alloc(64, v28);
		    v26 = (int8_t *)v29;
		} else {
		    v26 = v4;
		}
		int8_t *v30 = v26;
		int64_t v31 = (int64_t)v8;
		v5[v8] = v16;
		int64_t v32 = estimate_all(v1, v2, v3, v30 + v31, v5);
		if (v32 == v6) {
		    v5[v8] = v9;
		}
		if (v25) {
		    free(v30);
		}
		v24 = v32;
	    }
	    v23 = v24;
	}
	v15 = v23;
    } else {
	v15 = v14;
    }
    return v15;
}
