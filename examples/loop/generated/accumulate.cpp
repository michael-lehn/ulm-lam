#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
extern "C" void
foo(size_t, double *);
/* accumulate: matrices by reference (row-major), owned by the caller
 *   v1  double[500][500]  inout  may be overwritten (unspecified on exit)
 *   v2  double[500][500]  inout  may be overwritten (unspecified on exit)
 *   v3  double[500][500]  inout  updated in place
 *   v4  size_t            in
 *   v5  int8_t *          work   workspace (bytes)
 *   v6  int64_t *         info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *    -6  v6: workspace too small
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
/* accumulate_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[500][500]  in     read only
 *   v2  double[500][500]  in     read only
 *   v3  double[500][500]  inout  updated in place
 *   v4  size_t            in
 *   v5  int8_t *          work   workspace (bytes)
 *   v6  int64_t *         info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *    -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
accumulate_keep(double *v1, double *v2, double *v3, size_t v4, int8_t *v5,
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
    v6[v9] = v8;
    int64_t v17 = accumulate(v12, v14, v16, v4, v5, v6);
    int64_t v18;
    if (v17 == v7) {
	int64_t v19 = v6[v9];
	uint64_t v20 = (uint64_t)v19;
	uint64_t v21 = (uint64_t)63;
	int64_t v22 = (int64_t)(v20 + v21);
	uint64_t v23 = (uint64_t)v22;
	uint64_t v24 = (uint64_t)-64;
	int64_t v25 = (int64_t)(v23 & v24);
	v6[v9] = v10;
	int64_t v26;
	if (v10 == v8) {
	    v6[v9] = v25;
	    v26 = v7;
	} else {
	    int64_t v27;
	    if (v10 > v7 & v10 < v25) {
		v27 = -6;
	    } else {
		bool v28 = v10 == v7;
		int8_t *v29;
		if (v28) {
		    ptrdiff_t v30 = (ptrdiff_t)v25;
		    size_t v31 = (size_t)v30;
		    void *v32 = aligned_alloc(64, v31);
		    v29 = (int8_t *)v32;
		} else {
		    v29 = v5;
		}
		int8_t *v33 = v29;
		int64_t v34 = (int64_t)v9;
		v6[v9] = v19;
		int64_t v35 = accumulate(v1, v2, v3, v4, v33 + v34, v6);
		if (v35 == v7) {
		    v6[v9] = v10;
		}
		if (v28) {
		    free(v33);
		}
		v27 = v35;
	    }
	    v26 = v27;
	}
	v18 = v26;
    } else {
	v18 = v17;
    }
    return v18;
}
