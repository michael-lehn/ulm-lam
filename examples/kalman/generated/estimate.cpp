#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/ulmblas.h>
template <typename T>
static void
ulm_lam_potrf(bool lower, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
              std::ptrdiff_t incCol, bool checked)
{
    std::ptrdiff_t info = ulmBLAS::potrf(lower, n, A, incRow, incCol);
    if (checked && info != 0) {
	std::fprintf(
	    stderr,
	    "ulm_lam: matrix is not positive definite (potrf info %td)\n",
	    info);
	std::abort();
    }
}
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
observe(size_t, double *);
extern "C" void
record(size_t, double *);
/* estimate: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
estimate(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 35537600;
    double v7 = 1.00000000000000000e+00;
    double v8 = 0.0e+00;
    int64_t v9 = 0;
    size_t v10 = 0;
    int64_t v11 = v5[v10];
    if (v11 == -1) {
	v5[v10] = v6;
    } else {
	if (v11 > v9 & v11 < v6) {
	    ulm_lam_workspace_error(v6, v11);
	} else {
	    bool v12 = v11 == v9;
	    int8_t *v13;
	    if (v12) {
		void *v14 = aligned_alloc(64, 35537600);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v10;
	    double *v17 = (double *)(v15 + v16);
	    memcpy(v17, v2, 32000000);
	    ulm_lam_potrf(true, 2000, v17, 2000, 1, true);
	    int64_t v18 = (int64_t)32000000;
	    double *v19 = (double *)(v15 + v18);
	    int64_t v20 = (int64_t)32320000;
	    double *v21 = (double *)(v15 + v20);
	    memcpy(v21, v1, 3200000);
	    ulmBLAS::trlsm<std::ptrdiff_t>(2000, 200, v7, false, false, v17,
	                                   2000, 1, v21, 200, 1);
	    ulmBLAS::gemm<std::ptrdiff_t>(200, 200, 2000, v7, false, v21, 1,
	                                  200, false, v21, 200, 1, v8, v19, 200,
	                                  1);
	    ulm_lam_potrf(true, 200, v19, 200, 1, true);
	    int64_t v22 = (int64_t)35520000;
	    double *v23 = (double *)(v15 + v22);
	    int64_t v24 = (int64_t)35536000;
	    double *v25 = (double *)(v15 + v24);
	    for (size_t i26 = v10; i26 < v3; i26 += 1) {
		observe(i26, v23);
		ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v17, 2000, 1, v23,
		                               1);
		ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v7, v21, 1, 200, v23,
		                              1, v8, v25, 1);
		ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v19, 200, 1, v25, 1);
		ulmBLAS::trusv<std::ptrdiff_t>(200, false, v19, 1, 200, v25, 1);
		record(i26, v25);
	    }
	    if (v12) {
		free(v15);
	    }
	}
    }
    return;
}
