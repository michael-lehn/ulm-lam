#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
extern "C" void
foo(size_t, double *);
/* accumulate: matrices by reference (row-major), owned by the caller
 *   v1  double[500][500]  in     read only
 *   v2  double[500][500]  in     read only
 *   v3  double[500][500]  inout  updated in place
 *   v4  size_t            in
 *   v5  int8_t *          work   workspace (bytes)
 *   v6  int64_t *         info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *    0  success
 *   -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
accumulate(double *v1, double *v2, double *v3, size_t v4, int8_t *v5,
           int64_t *v6)
{
    int64_t v7 = 4000000;
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
	bool v14 = v12 > v10 & v12 < v7;
	if (v14) {
	} else {
	    bool v15 = v12 == v10;
	    int8_t *v16;
	    if (v15) {
		void *v17 = aligned_alloc(64, 4000000);
		v16 = (int8_t *)v17;
	    } else {
		v16 = v5;
	    }
	    int8_t *v18 = v16;
	    int64_t v19 = (int64_t)v11;
	    double *v20 = (double *)(v18 + v19);
	    int64_t v21 = (int64_t)2000000;
	    double *v22 = (double *)(v18 + v21);
	    for (size_t i23 = v11; i23 < v4; i23 += 1) {
		foo(i23, v20);
		ulmBLAS::gemm<std::ptrdiff_t>(500, 500, 500, v8, false, v1, 500,
		                              1, false, v20, 500, 1, v9, v22,
		                              500, 1);
		ulmBLAS::gemm<std::ptrdiff_t>(500, 500, 500, v8, false, v22,
		                              500, 1, false, v2, 500, 1, v8, v3,
		                              500, 1);
	    }
	    if (v15) {
		free(v18);
	    }
	}
	v13 = v14 ? -6 : v10;
    }
    return v13;
}
