#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
/* attention: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  size_t          in
 *   v4  double[v1][v2]  in    read only
 *   v5  double[v1][v2]  in    read only
 *   v6  double[v1][v3]  in    read only
 *   v7  double[v1][v3]  out   written
 *   v8  int8_t *        work  workspace (bytes)
 *   v9  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *    -9  v9: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
attention(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double *v6,
          double *v7, int8_t *v8, int64_t *v9)
{
    double v10 = 1.00000000000000000e+00;
    double v11 = 0.0e+00;
    int64_t v12 = 0;
    size_t v13 = 0;
    int64_t v14 = v9[v13];
    bool v15 = ((v1 * v2) * v1) * 2 + ((v1 * v1) * v3) * 2 <=
               ((v1 * v2) * v3) * 2 + ((v1 * v2) * v3) * 2;
    size_t v16 = v15 ? (v1 * 8) * v1 + 63 & -64 : (v2 * 8) * v3 + 63 & -64;
    ptrdiff_t v17 = (ptrdiff_t)v16;
    int64_t v18 = (int64_t)v17;
    int64_t v19;
    if (v14 == -1) {
	v9[v13] = v18;
	v19 = v12;
    } else {
	bool v20 = v14 > v12 & v14 < v18;
	if (v20) {
	} else {
	    bool v21 = v14 == v12;
	    int8_t *v22;
	    if (v21) {
		void *v23 = aligned_alloc(64, v16);
		v22 = (int8_t *)v23;
	    } else {
		v22 = v8;
	    }
	    int8_t *v24 = v22;
	    if (v15) {
		int64_t v25 = (int64_t)v13;
		double *v26 = (double *)(v24 + v25);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v1, v2, v10, false, v4, v2, 1,
		                              false, v5, 1, v2, v11, v26, v1,
		                              1);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v3, v1, v10, false, v26, v1,
		                              1, false, v6, v3, 1, v11, v7, v3,
		                              1);
	    } else {
		int64_t v27 = (int64_t)v13;
		double *v28 = (double *)(v24 + v27);
		ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false, v5, 1, v2,
		                              false, v6, v3, 1, v11, v28, v3,
		                              1);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v3, v2, v10, false, v4, v2, 1,
		                              false, v28, v3, 1, v11, v7, v3,
		                              1);
	    }
	    if (v21) {
		free(v24);
	    }
	}
	v19 = v20 ? -9 : v12;
    }
    return v19;
}
