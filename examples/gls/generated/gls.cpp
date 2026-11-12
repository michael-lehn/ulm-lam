#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/trtrs.h>
#include <ulmblas/ulmblas.h>
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
