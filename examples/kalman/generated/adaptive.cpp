#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/ulmblas.h>
template <typename T>
static void
ulm_lam_potrf(bool lower, int n, T *A, int incRow, int incCol, bool checked)
{
    int info = ulmBLAS::potrf(lower, n, A, incRow, incCol);
    if (checked && info != 0) {
	std::fprintf(
	    stderr,
	    "ulm_lam: matrix is not positive definite (potrf info %d)\n", info);
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
update_covariance(size_t, double *);
extern "C" void
observe(size_t, double *);
extern "C" void
record(size_t, double *);
/* estimate_adaptive: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
estimate_adaptive(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    double v6 = 1.00000000000000000e+00;
    double v7 = 0.0e+00;
    int64_t v8 = 0;
    int64_t v9 = 67537600;
    size_t v10 = 0;
    int64_t v11 = v5[v10];
    if (v11 == -1) {
	v5[v10] = v9;
    } else {
	if (v11 > v8 & v11 < v9) {
	    ulm_lam_workspace_error(v9, v11);
	} else {
	    bool v12 = v11 == v8;
	    int8_t *v13;
	    if (v12) {
		void *v14 = aligned_alloc(64, 67537600);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v10;
	    double *v17 = (double *)(v15 + v16);
	    memcpy(v17, v2, 32000000);
	    int64_t v18 = (int64_t)32000000;
	    double *v19 = (double *)(v15 + v18);
	    int64_t v20 = (int64_t)64000000;
	    double *v21 = (double *)(v15 + v20);
	    int64_t v22 = (int64_t)64016000;
	    double *v23 = (double *)(v15 + v22);
	    int64_t v24 = (int64_t)64336000;
	    double *v25 = (double *)(v15 + v24);
	    int64_t v26 = (int64_t)67536000;
	    double *v27 = (double *)(v15 + v26);
	    for (size_t i28 = v10; i28 < v3; i28 += 1) {
		update_covariance(i28, v17);
		memcpy(v19, v17, 32000000);
		ulm_lam_potrf(true, 2000, v19, 2000, 1, true);
		observe(i28, v21);
		memcpy(v25, v1, 3200000);
		ulmBLAS::trlsm(2000, 200, v6, false, false, v19, 2000, 1, v25,
		               200, 1);
		ulmBLAS::gemm(200, 200, 2000, v6, false, v25, 1, 200, false,
		              v25, 200, 1, v7, v23, 200, 1);
		ulm_lam_potrf(true, 200, v23, 200, 1, true);
		ulmBLAS::trlsv(2000, false, v19, 2000, 1, v21, 1);
		ulmBLAS::gemv(200, 2000, v6, v25, 1, 200, v21, 1, v7, v27, 1);
		ulmBLAS::trlsv(200, false, v23, 200, 1, v27, 1);
		ulmBLAS::trusv(200, false, v23, 1, 200, v27, 1);
		record(i28, v27);
	    }
	    if (v12) {
		free(v15);
	    }
	}
    }
    return;
}
