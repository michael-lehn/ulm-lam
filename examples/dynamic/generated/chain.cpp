#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
/* chain: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   size_t          in
 *   v4   size_t          in
 *   v5   double[v1][v2]  in    read only
 *   v6   double[v2][v3]  in    read only
 *   v7   double[v3][v4]  in    read only
 *   v8   double[v1][v4]  out   written
 *   v9   int8_t *        work  workspace (bytes)
 *   v10  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *    0  success
 *   -10  v10: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
chain(size_t v1, size_t v2, size_t v3, size_t v4, double *v5, double *v6,
      double *v7, double *v8, int8_t *v9, int64_t *v10)
{
    double v11 = 1.00000000000000000e+00;
    double v12 = 0.0e+00;
    int64_t v13 = 0;
    size_t v14 = 0;
    int64_t v15 = v10[v14];
    bool v16 = ((v1 * v2) * v3) * 2 + ((v1 * v3) * v4) * 2 <=
               ((v2 * v3) * v4) * 2 + ((v1 * v2) * v4) * 2;
    size_t v17 = v16 ? (v1 * 8) * v3 + 63 & -64 : (v2 * 8) * v4 + 63 & -64;
    ptrdiff_t v18 = (ptrdiff_t)v17;
    int64_t v19 = (int64_t)v18;
    int64_t v20;
    if (v15 == -1) {
	v10[v14] = v19;
	v20 = v13;
    } else {
	bool v21 = v15 > v13 & v15 < v19;
	if (v21) {
	} else {
	    bool v22 = v15 == v13;
	    int8_t *v23;
	    if (v22) {
		void *v24 = aligned_alloc(64, v17);
		v23 = (int8_t *)v24;
	    } else {
		v23 = v9;
	    }
	    int8_t *v25 = v23;
	    if (v16) {
		int64_t v26 = (int64_t)v14;
		double *v27 = (double *)(v25 + v26);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v3, v2, v11, false, v5, v2, 1,
		                              false, v6, v3, 1, v12, v27, v3,
		                              1);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v4, v3, v11, false, v27, v3,
		                              1, false, v7, v4, 1, v12, v8, v4,
		                              1);
	    } else {
		int64_t v28 = (int64_t)v14;
		double *v29 = (double *)(v25 + v28);
		ulmBLAS::gemm<std::ptrdiff_t>(v2, v4, v3, v11, false, v6, v3, 1,
		                              false, v7, v4, 1, v12, v29, v4,
		                              1);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v4, v2, v11, false, v5, v2, 1,
		                              false, v29, v4, 1, v12, v8, v4,
		                              1);
	    }
	    if (v22) {
		free(v25);
	    }
	}
	v20 = v21 ? -10 : v13;
    }
    return v20;
}
