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
 *   v5   double[v1][v2]  inout  may be overwritten (unspecified on exit)
 *   v6   double[v2][v3]  inout  may be overwritten (unspecified on exit)
 *   v7   double[v3][v4]  inout  may be overwritten (unspecified on exit)
 *   v8   double[v1][v4]  out    written
 *   v9   int8_t *        work   workspace (bytes)
 *   v10  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *    -10  v10: workspace too small
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
/* chain_keep: matrices by reference (row-major), owned by the caller
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
 *      0  success
 *    -10  v10: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
chain_keep(size_t v1, size_t v2, size_t v3, size_t v4, double *v5, double *v6,
           double *v7, double *v8, int8_t *v9, int64_t *v10)
{
    int64_t v11 = 0;
    int64_t v12 = -1;
    size_t v13 = 0;
    int64_t v14 = v10[v13];
    int64_t v15 = (int64_t)4;
    double *v16 = (double *)(v9 + v15);
    int64_t v17 = (int64_t)5;
    double *v18 = (double *)(v9 + v17);
    int64_t v19 = (int64_t)6;
    double *v20 = (double *)(v9 + v19);
    int64_t v21 = (int64_t)7;
    double *v22 = (double *)(v9 + v21);
    v10[v13] = v12;
    int64_t v23 = chain(v1, v2, v3, v4, v16, v18, v20, v22, v9, v10);
    int64_t v24;
    if (v23 == v11) {
	int64_t v25 = v10[v13];
	uint64_t v26 = (uint64_t)v25;
	uint64_t v27 = (uint64_t)63;
	int64_t v28 = (int64_t)(v26 + v27);
	uint64_t v29 = (uint64_t)v28;
	uint64_t v30 = (uint64_t)-64;
	int64_t v31 = (int64_t)(v29 & v30);
	v10[v13] = v14;
	int64_t v32;
	if (v14 == v12) {
	    v10[v13] = v31;
	    v32 = v11;
	} else {
	    int64_t v33;
	    if (v14 > v11 & v14 < v31) {
		v33 = -10;
	    } else {
		bool v34 = v14 == v11;
		int8_t *v35;
		if (v34) {
		    ptrdiff_t v36 = (ptrdiff_t)v31;
		    size_t v37 = (size_t)v36;
		    void *v38 = aligned_alloc(64, v37);
		    v35 = (int8_t *)v38;
		} else {
		    v35 = v9;
		}
		int8_t *v39 = v35;
		int64_t v40 = (int64_t)v13;
		v10[v13] = v25;
		int64_t v41 =
		    chain(v1, v2, v3, v4, v5, v6, v7, v8, v39 + v40, v10);
		if (v41 == v11) {
		    v10[v13] = v14;
		}
		if (v34) {
		    free(v39);
		}
		v33 = v41;
	    }
	    v32 = v33;
	}
	v24 = v32;
    } else {
	v24 = v23;
    }
    return v24;
}
