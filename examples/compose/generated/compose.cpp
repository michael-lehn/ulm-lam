#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/ulmblas.h>
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
 *    0  success
 *    1  Cholesky factorisation of v2: not positive definite
 *    2  Cholesky factorisation of a computed matrix: not positive definite
 *   -6  v6: workspace too small
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
	    memcpy(v20, v2, 32000000);
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
		memcpy(v29, v1, 3200000);
		ulmBLAS::trlsm<std::ptrdiff_t>(2000, 200, v8, false, false, v20,
		                               2000, 1, v29, 200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v8, v29, 1, 200, v9,
		                               v27, 200, 1);
		int64_t v30 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v27, 200, 1);
		int64_t v31 = v30 != v10 ? 2 : v10;
		if (v31 == v10) {
		    int64_t v32 = (int64_t)35520000;
		    double *v33 = (double *)(v18 + v32);
		    memcpy(v33, v3, 16000);
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
 *    0  success
 *    1  call of gls returned info != 0
 *   -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate_all(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    int64_t v7 = -64;
    int64_t v8 = 63;
    int64_t v9 = -1;
    size_t v10 = 0;
    int64_t v11 = v5[v10];
    int64_t v12 = (int64_t)v10;
    double *v13 = (double *)(v4 + v12);
    int64_t v14 = (int64_t)v10;
    double *v15 = (double *)(v4 + v14);
    int64_t v16 = (int64_t)v10;
    double *v17 = (double *)(v4 + v16);
    int64_t v18 = (int64_t)v10;
    double *v19 = (double *)(v4 + v18);
    v5[v10] = v9;
    int64_t v20 = gls(v13, v15, v17, v19, v4, v5);
    int64_t v21;
    if (v20 == v6) {
	int64_t v22 = v5[v10];
	uint64_t v23 = (uint64_t)v22;
	uint64_t v24 = (uint64_t)v8;
	int64_t v25 = (int64_t)(v23 + v24);
	uint64_t v26 = (uint64_t)v25;
	uint64_t v27 = (uint64_t)v7;
	int64_t v28 = (int64_t)(v26 & v27);
	uint64_t v29 = (uint64_t)v28;
	uint64_t v30 = (uint64_t)17600;
	int64_t v31 = (int64_t)(v29 + v30);
	v5[v10] = v9;
	postprocess(v10, v19, v4, v5);
	int64_t v32 = v5[v10];
	uint64_t v33 = (uint64_t)v32;
	uint64_t v34 = (uint64_t)v8;
	int64_t v35 = (int64_t)(v33 + v34);
	uint64_t v36 = (uint64_t)v35;
	uint64_t v37 = (uint64_t)v7;
	int64_t v38 = (int64_t)(v36 & v37);
	uint64_t v39 = (uint64_t)v31;
	uint64_t v40 = (uint64_t)v38;
	int64_t v41 = (int64_t)(v39 + v40);
	v5[v10] = v11;
	int64_t v42;
	if (v11 == v9) {
	    v5[v10] = v41;
	    v42 = v6;
	} else {
	    int64_t v43;
	    if (v11 > v6 & v11 < v41) {
		v43 = -5;
	    } else {
		bool v44 = v11 == v6;
		int8_t *v45;
		if (v44) {
		    ptrdiff_t v46 = (ptrdiff_t)v41;
		    size_t v47 = (size_t)v46;
		    void *v48 = aligned_alloc(64, v47);
		    v45 = (int8_t *)v48;
		} else {
		    v45 = v4;
		}
		int8_t *v49 = v45;
		int64_t v50 = (int64_t)v10;
		double *v51 = (double *)(v49 + v50);
		int64_t v52 = (int64_t)16000;
		double *v53 = (double *)(v49 + v52);
		int64_t v54;
		v54 = v6;
		for (size_t i55 = v10; i55 < v3; i55 += 1) {
		    int64_t v56 = v54;
		    int64_t v57;
		    if (v56 == v6) {
			observe(i55, v51);
			int64_t v58 = (int64_t)17600;
			v5[v10] = v22;
			bool v59 =
			    (bool)(gls(v1, v2, v51, v53, v49 + v58, v5) != v6);
			uint64_t v60 = (uint64_t)v59;
			int64_t v61 = (int64_t)v60;
			if (v61 == v6) {
			    ptrdiff_t v62 = (ptrdiff_t)v31;
			    size_t v63 = (size_t)v62;
			    int64_t v64 = (int64_t)v63;
			    int8_t *v65 = v49 + v64;
			    v5[v10] = v32;
			    postprocess(i55, v53, v65, v5);
			}
			v57 = v61;
		    } else {
			v57 = v56;
		    }
		    v54 = v57;
		}
		int64_t v66 = v54;
		if (v66 == v6) {
		    v5[v10] = v11;
		}
		if (v44) {
		    free(v49);
		}
		v43 = v66;
	    }
	    v42 = v43;
	}
	v21 = v42;
    } else {
	v21 = v20;
    }
    return v21;
}
