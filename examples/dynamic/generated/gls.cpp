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
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  double[v1][v2]  in    read only
 *   v4  double[v1][v1]  in    read only
 *   v5  double[v1][1]   in    read only
 *   v6  double[v2][1]   out   written
 *   v7  int8_t *        work  workspace (bytes)
 *   v8  int64_t *       info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
gls(size_t v1, size_t v2, double *v3, double *v4, double *v5, double *v6,
    int8_t *v7, int64_t *v8)
{
    double v9 = 1.00000000000000000e+00;
    double v10 = 0.0e+00;
    int64_t v11 = 0;
    size_t v12 = -64;
    size_t v13 = 63;
    size_t v14 = 8;
    size_t v15 = 0;
    int64_t v16 = v8[v15];
    size_t v17 = v1 * v14;
    size_t v18 = v17 * v1 + v13 & v12;
    size_t v19 = v18 + ((v2 * v14) * v2 + v13 & v12);
    size_t v20 = v19 + (v17 * v2 + v13 & v12);
    size_t v21 = v20 + (v17 + v13 & v12);
    ptrdiff_t v22 = (ptrdiff_t)v21;
    int64_t v23 = (int64_t)v22;
    if (v16 == -1) {
	v8[v15] = v23;
    } else {
	if (v16 > v11 & v16 < v23) {
	    ulm_lam_workspace_error(v23, v16);
	} else {
	    bool v24 = v16 == v11;
	    int8_t *v25;
	    if (v24) {
		void *v26 = aligned_alloc(64, v21);
		v25 = (int8_t *)v26;
	    } else {
		v25 = v7;
	    }
	    int8_t *v27 = v25;
	    int64_t v28 = (int64_t)v15;
	    double *v29 = (double *)(v27 + v28);
	    size_t v30 = (8 * v1) * v1;
	    memcpy(v29, v4, v30);
	    ulm_lam_potrf(true, v1, v29, v1, 1, true);
	    int64_t v31 = (int64_t)v18;
	    double *v32 = (double *)(v27 + v31);
	    int64_t v33 = (int64_t)v19;
	    double *v34 = (double *)(v27 + v33);
	    size_t v35 = (8 * v1) * v2;
	    memcpy(v34, v3, v35);
	    ulmBLAS::trlsm<std::ptrdiff_t>(v1, v2, v9, false, false, v29, v1, 1,
	                                   v34, v2, 1);
	    ulmBLAS::gemm<std::ptrdiff_t>(v2, v2, v1, v9, false, v34, 1, v2,
	                                  false, v34, v2, 1, v10, v32, v2, 1);
	    ulm_lam_potrf(true, v2, v32, v2, 1, true);
	    int64_t v36 = (int64_t)v20;
	    double *v37 = (double *)(v27 + v36);
	    size_t v38 = 8 * v1;
	    memcpy(v37, v5, v38);
	    ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v29, v1, 1, v37, 1);
	    ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v9, v34, 1, v2, v37, 1, v10,
	                                  v6, 1);
	    ulmBLAS::trlsv<std::ptrdiff_t>(v2, false, v32, v2, 1, v6, 1);
	    ulmBLAS::trusv<std::ptrdiff_t>(v2, false, v32, 1, v2, v6, 1);
	    if (v24) {
		free(v27);
	    }
	}
    }
    return;
}
