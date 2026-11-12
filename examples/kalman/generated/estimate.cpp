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
record(size_t, double *);
/* estimate: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *    0  success
 *    1  Cholesky factorisation of v2: not positive definite
 *    2  Cholesky factorisation of a computed matrix: not positive definite
 *   -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
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
	    memcpy(v19, v2, 32000000);
	    int64_t v20 =
	        ulmBLAS::potrf<std::ptrdiff_t>(true, 2000, v19, 2000, 1);
	    bool v21 = (bool)(v20 != v9);
	    uint64_t v22 = (uint64_t)v21;
	    int64_t v23 = (int64_t)v22;
	    int64_t v24;
	    if (v23 == v9) {
		int64_t v25 = (int64_t)32000000;
		double *v26 = (double *)(v17 + v25);
		int64_t v27 = (int64_t)32320000;
		double *v28 = (double *)(v17 + v27);
		memcpy(v28, v1, 3200000);
		ulmBLAS::trlsm<std::ptrdiff_t>(2000, 200, v7, false, false, v19,
		                               2000, 1, v28, 200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v7, v28, 1, 200, v8,
		                               v26, 200, 1);
		int64_t v29 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v26, 200, 1);
		int64_t v30 = v29 != v9 ? 2 : v9;
		if (v30 == v9) {
		    int64_t v31 = (int64_t)35520000;
		    double *v32 = (double *)(v17 + v31);
		    int64_t v33 = (int64_t)35536000;
		    double *v34 = (double *)(v17 + v33);
		    for (size_t i35 = v10; i35 < v3; i35 += 1) {
			observe(i35, v32);
			ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v19, 2000,
			                               1, v32, 1);
			ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v7, v28, 1,
			                              200, v32, 1, v8, v34, 1);
			ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v26, 200, 1,
			                               v34, 1);
			ulmBLAS::trusv<std::ptrdiff_t>(200, false, v26, 1, 200,
			                               v34, 1);
			record(i35, v34);
		    }
		}
		v24 = v30;
	    } else {
		v24 = v23;
	    }
	    int64_t v36 = v24;
	    if (v14) {
		free(v17);
	    }
	    v13 = v36;
	}
	v12 = v13;
    }
    return v12;
}
