#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrs.h>
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
template <typename T>
static void
ulm_lam_getrf(std::ptrdiff_t m, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
              std::ptrdiff_t incCol, int32_t *piv, bool checked)
{
    std::ptrdiff_t info =
        ulmBLAS::getrf(m, n, A, incRow, incCol, piv, std::ptrdiff_t(1));
    if (checked && info != 0) {
	std::fprintf(stderr, "ulm_lam: matrix is singular (getrf info %td)\n",
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
/* lu: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][32]  in    read only
 *   v3  double[64][32]  out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
lu(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 33024;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    if (v9 == -1) {
	v5[v8] = v6;
    } else {
	if (v9 > v7 & v9 < v6) {
	    ulm_lam_workspace_error(v6, v9);
	} else {
	    bool v10 = v9 == v7;
	    int8_t *v11;
	    if (v10) {
		void *v12 = aligned_alloc(64, 33024);
		v11 = (int8_t *)v12;
	    } else {
		v11 = v4;
	    }
	    int8_t *v13 = v11;
	    int64_t v14 = (int64_t)v8;
	    int32_t *v15 = (int32_t *)(v13 + v14);
	    int64_t v16 = (int64_t)256;
	    double *v17 = (double *)(v13 + v16);
	    memcpy(v17, v1, 32768);
	    ulm_lam_getrf(64, 64, v17, 64, 1, v15, true);
	    memcpy(v3, v2, 16384);
	    ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v17, 64, 1,
	                                   v15, 1, v3, 32, 1);
	    if (v10) {
		free(v13);
	    }
	}
    }
    return;
}
/* lu_t: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][32]  in    read only
 *   v3  double[64][32]  out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
lu_t(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 33024;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    if (v9 == -1) {
	v5[v8] = v6;
    } else {
	if (v9 > v7 & v9 < v6) {
	    ulm_lam_workspace_error(v6, v9);
	} else {
	    bool v10 = v9 == v7;
	    int8_t *v11;
	    if (v10) {
		void *v12 = aligned_alloc(64, 33024);
		v11 = (int8_t *)v12;
	    } else {
		v11 = v4;
	    }
	    int8_t *v13 = v11;
	    int64_t v14 = (int64_t)v8;
	    int32_t *v15 = (int32_t *)(v13 + v14);
	    int64_t v16 = (int64_t)256;
	    double *v17 = (double *)(v13 + v16);
	    memcpy(v17, v1, 32768);
	    ulm_lam_getrf(64, 64, v17, 64, 1, v15, true);
	    memcpy(v3, v2, 16384);
	    ulmBLAS::getrs<std::ptrdiff_t>(true, false, 64, 32, v17, 64, 1, v15,
	                                   1, v3, 32, 1);
	    if (v10) {
		free(v13);
	    }
	}
    }
    return;
}
/* upper: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][32]  in   read only
 *   v3  double[64][32]  out  written
 */
extern "C" void
upper(double *v1, double *v2, double *v3)
{
    double v4 = 1.00000000000000000e+00;
    memcpy(v3, v2, 16384);
    ulmBLAS::trusm<std::ptrdiff_t>(64, 32, v4, false, false, v1, 64, 1, v3, 32,
                                   1);
    return;
}
/* right: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 */
extern "C" void
right(double *v1, double *v2, double *v3)
{
    double v4 = 1.00000000000000000e+00;
    memcpy(v3, v1, 16384);
    ulmBLAS::trusm<std::ptrdiff_t>(64, 32, v4, false, false, v2, 1, 64, v3, 1,
                                   64);
    return;
}
/* right_t: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 */
extern "C" void
right_t(double *v1, double *v2, double *v3)
{
    double v4 = 1.00000000000000000e+00;
    memcpy(v3, v1, 16384);
    ulmBLAS::trlsm<std::ptrdiff_t>(64, 32, v4, false, false, v2, 64, 1, v3, 1,
                                   64);
    return;
}
/* chol_vec: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][1]   in    read only
 *   v3  double[64][1]   out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
chol_vec(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 32768;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    if (v9 == -1) {
	v5[v8] = v6;
    } else {
	if (v9 > v7 & v9 < v6) {
	    ulm_lam_workspace_error(v6, v9);
	} else {
	    bool v10 = v9 == v7;
	    int8_t *v11;
	    if (v10) {
		void *v12 = aligned_alloc(64, 32768);
		v11 = (int8_t *)v12;
	    } else {
		v11 = v4;
	    }
	    int8_t *v13 = v11;
	    int64_t v14 = (int64_t)v8;
	    double *v15 = (double *)(v13 + v14);
	    memcpy(v15, v1, 32768);
	    ulm_lam_potrf(true, 64, v15, 64, 1, true);
	    memcpy(v3, v2, 512);
	    ulmBLAS::trlsv<std::ptrdiff_t>(64, false, v15, 64, 1, v3, 1);
	    ulmBLAS::trusv<std::ptrdiff_t>(64, false, v15, 1, 64, v3, 1);
	    if (v10) {
		free(v13);
	    }
	}
    }
    return;
}
/* sum: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[64][64]  in   read only
 *   v4  double[64][64]  out  written
 */
extern "C" void
sum(double *v1, double *v2, double *v3, double *v4)
{
    double v5 = 2.00000000000000000e+00;
    double v6 = -1.00000000000000000e+00;
    memcpy(v4, v1, 32768);
    ulmBLAS::geaxpy<std::ptrdiff_t>(64, 64, v5, v2, 64, 1, v4, 64, 1);
    ulmBLAS::geaxpy<std::ptrdiff_t>(64, 64, v6, v3, 64, 1, v4, 64, 1);
    return;
}
/* mixed: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][64]  in    read only
 *   v3  double[64][32]  in    read only
 *   v4  double[64][32]  in    read only
 *   v5  double[64][32]  out   written
 *   v6  int8_t *        work  workspace (bytes)
 *   v7  int64_t *       info  -1: query, 0: allocate, >0: size of work
 */
extern "C" void
mixed(double *v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
      int64_t *v7)
{
    int64_t v8 = 49408;
    double v9 = 1.00000000000000000e+00;
    double v10 = 0.0e+00;
    int64_t v11 = 0;
    size_t v12 = 0;
    int64_t v13 = v7[v12];
    if (v13 == -1) {
	v7[v12] = v8;
    } else {
	if (v13 > v11 & v13 < v8) {
	    ulm_lam_workspace_error(v8, v13);
	} else {
	    bool v14 = v13 == v11;
	    int8_t *v15;
	    if (v14) {
		void *v16 = aligned_alloc(64, 49408);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v6;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 = (int64_t)v12;
	    int32_t *v19 = (int32_t *)(v17 + v18);
	    int64_t v20 = (int64_t)256;
	    double *v21 = (double *)(v17 + v20);
	    memcpy(v21, v1, 32768);
	    ulm_lam_getrf(64, 64, v21, 64, 1, v19, true);
	    int64_t v22 = (int64_t)33024;
	    double *v23 = (double *)(v17 + v22);
	    ulmBLAS::gemm<std::ptrdiff_t>(64, 32, 64, v9, false, v2, 64, 1,
	                                  false, v3, 32, 1, v10, v23, 32, 1);
	    ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v21, 64, 1,
	                                   v19, 1, v23, 32, 1);
	    memcpy(v5, v4, 16384);
	    ulmBLAS::geaxpy<std::ptrdiff_t>(64, 32, v9, v23, 32, 1, v5, 32, 1);
	    if (v14) {
		free(v17);
	    }
	}
    }
    return;
}
