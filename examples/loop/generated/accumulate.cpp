#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
extern "C" __attribute__((weak)) void
ulm_lam_workspace_error(int64_t required, int64_t given)
{
    std::fprintf(
        stderr,
        "ulm_lam: workspace too small: %lld bytes required, %lld given\n",
        (long long)required, (long long)given);
    std::abort();
}
extern "C" void
ulm_lam_workspace_error(int64_t, int64_t);
extern "C" void
foo(size_t, double *);
/* accumulate: matrices by reference (row-major), owned by the caller
 *   v1  double[500][500]  in     read only
 *   v2  double[500][500]  in     read only
 *   v3  double[500][500]  inout  updated in place
 *   v4  size_t            in
 *   v5  int8_t *          work   workspace (bytes)
 *   v6  int64_t *         info   -1: query, 0: allocate, >0: size of work
 */
extern "C" void
accumulate(double *v1, double *v2, double *v3, size_t v4, int8_t *v5,
           int64_t *v6)
{
    int64_t v7 = 4000000;
    double v8 = 1.00000000000000000e+00;
    double v9 = 0.0e+00;
    int64_t v10 = 0;
    size_t v11 = 0;
    int64_t v12 = v6[v11];
    if (v12 == -1) {
	v6[v11] = v7;
    } else {
	if (v12 > v10 & v12 < v7) {
	    ulm_lam_workspace_error(v7, v12);
	} else {
	    bool v13 = v12 == v10;
	    int8_t *v14;
	    if (v13) {
		void *v15 = aligned_alloc(64, 4000000);
		v14 = (int8_t *)v15;
	    } else {
		v14 = v5;
	    }
	    int8_t *v16 = v14;
	    int64_t v17 = (int64_t)v11;
	    double *v18 = (double *)(v16 + v17);
	    int64_t v19 = (int64_t)2000000;
	    double *v20 = (double *)(v16 + v19);
	    for (size_t i21 = v11; i21 < v4; i21 += 1) {
		foo(i21, v18);
		ulmBLAS::gemm<std::ptrdiff_t>(500, 500, 500, v8, false, v1, 500,
		                              1, false, v18, 500, 1, v9, v20,
		                              500, 1);
		ulmBLAS::gemm<std::ptrdiff_t>(500, 500, 500, v8, false, v20,
		                              500, 1, false, v2, 500, 1, v8, v3,
		                              500, 1);
	    }
	    if (v13) {
		free(v16);
	    }
	}
    }
    return;
}
