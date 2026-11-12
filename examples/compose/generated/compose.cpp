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
gls(double *v1, double *v2, double *v3, double *v4, int8_t *v5, int64_t *v6)
{
    int64_t v7 = 35536000;
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
		void *v17 = aligned_alloc(64, 35536000);
		v16 = (int8_t *)v17;
	    } else {
		v16 = v5;
	    }
	    int8_t *v18 = v16;
	    int64_t v19 = (int64_t)v11;
	    double *v20 = (double *)(v18 + v19);
	    ulm_lam_copy(v20, v2, 32000000);
	    int64_t v21 =
	        ulmBLAS::potrf<std::ptrdiff_t>(true, 2000, v20, 2000, 1);
	    bool v22 = (bool)(v21 != v10);
	    uint64_t v23 = (uint64_t)v22;
	    int64_t v24 = (int64_t)v23;
	    int64_t v25;
	    if (v24 == v10) {
		int64_t v26 = (int64_t)32000000;
		double *v27 = (double *)(v18 + v26);
		int64_t v28 = (int64_t)32320000;
		double *v29 = (double *)(v18 + v28);
		ulm_lam_copy(v29, v1, 3200000);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false,
		                                     2000, 200, v20, 2000, 1,
		                                     v29, 200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v8, v29, 1, 200, v9,
		                               v27, 200, 1);
		int64_t v30 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v27, 200, 1);
		int64_t v31 = v30 != v10 ? 2 : v10;
		if (v31 == v10) {
		    int64_t v32 = (int64_t)35520000;
		    double *v33 = (double *)(v18 + v32);
		    ulm_lam_copy(v33, v3, 16000);
		    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v20, 2000, 1,
		                                   v33, 1);
		    ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v8, v29, 1, 200,
		                                  v33, 1, v9, v4, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v27, 200, 1, v4,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v27, 1, 200, v4,
		                                   1);
		}
		v25 = v31;
	    } else {
		v25 = v24;
	    }
	    int64_t v34 = v25;
	    if (v15) {
		free(v18);
	    }
	    v14 = v34;
	}
	v13 = v14;
    }
    return v13;
}
/* estimate_all: matrices by reference (row-major), owned by the caller
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
	uint64_t v31 = (uint64_t)17600;
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
		int64_t v57;
		v57 = v6;
		for (size_t i58 = v11; i58 < v3; i58 += v10) {
		    int64_t v59 = v57;
		    int64_t v60;
		    if (v59 == v6) {
			observe(i58, v54);
			int64_t v61 = (int64_t)17600;
			v5[v11] = v23;
			bool v62 =
			    (bool)(gls(v1, v2, v54, v56, v52 + v61, v5) != v6);
			uint64_t v63 = (uint64_t)v62;
			int64_t v64 = (int64_t)v63;
			if (v64 == v6) {
			    ptrdiff_t v65 = (ptrdiff_t)v32;
			    size_t v66 = (size_t)v65;
			    int64_t v67 = (int64_t)v66;
			    int8_t *v68 = v52 + v67;
			    v5[v11] = v35;
			    postprocess(i58, v56, v68, v5);
			}
			v60 = v64;
		    } else {
			v60 = v59;
		    }
		    v57 = v60;
		}
		int64_t v69 = v57;
		if (v69 == v6) {
		    v5[v11] = v12;
		}
		if (v47) {
		    free(v52);
		}
		v46 = v69;
	    }
	    v45 = v46;
	}
	v22 = v45;
    } else {
	v22 = v21;
    }
    return v22;
}
