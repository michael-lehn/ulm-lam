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
/* gls: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  double[2000][1]     in    read only
 *   v4  double[200][1]      out   written
 *   v5  int8_t *            work  workspace (bytes)
 *   v6  int64_t *           info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
gls(double *v1, double *v2, double *v3, double *v4, int8_t *v5, int64_t *v6)
{
    int64_t v7 = 35536000;
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
		void *v15 = aligned_alloc(64, 35536000);
		v14 = (int8_t *)v15;
	    } else {
		v14 = v5;
	    }
	    int8_t *v16 = v14;
	    int64_t v17 = (int64_t)v11;
	    double *v18 = (double *)(v16 + v17);
	    memcpy(v18, v2, 32000000);
	    ulm_lam_potrf(true, 2000, v18, 2000, 1, true);
	    int64_t v19 = (int64_t)32000000;
	    double *v20 = (double *)(v16 + v19);
	    int64_t v21 = (int64_t)32320000;
	    double *v22 = (double *)(v16 + v21);
	    memcpy(v22, v1, 3200000);
	    ulmBLAS::trlsm<std::ptrdiff_t>(2000, 200, v8, false, false, v18,
	                                   2000, 1, v22, 200, 1);
	    ulmBLAS::gemm<std::ptrdiff_t>(200, 200, 2000, v8, false, v22, 1,
	                                  200, false, v22, 200, 1, v9, v20, 200,
	                                  1);
	    ulm_lam_potrf(true, 200, v20, 200, 1, true);
	    int64_t v23 = (int64_t)35520000;
	    double *v24 = (double *)(v16 + v23);
	    memcpy(v24, v3, 16000);
	    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v18, 2000, 1, v24, 1);
	    ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v8, v22, 1, 200, v24, 1,
	                                  v9, v4, 1);
	    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v20, 200, 1, v4, 1);
	    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v20, 1, 200, v4, 1);
	    if (v13) {
		free(v16);
	    }
	}
    }
    return;
}
