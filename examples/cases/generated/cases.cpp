#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrs.h>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/trtrs.h>
#include <ulmblas/ulmblas.h>
static void
ulm_lam_copy(void *dst, const void *src, std::size_t bytes)
{
#ifdef _OPENMP
    const std::size_t chunk = std::size_t(1) << 20;
    if (bytes >= 8 * chunk) {
	std::ptrdiff_t chunks = (bytes + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
	for (std::ptrdiff_t c = 0; c < chunks; ++c) {
	    std::size_t off = c * chunk;
	    std::size_t len = bytes - off < chunk ? bytes - off : chunk;
	    std::memcpy(static_cast<char *>(dst) + off,
	                static_cast<const char *>(src) + off, len);
	}
	return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

/* lu: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][32]  in    read only
 *   v3  double[64][32]  out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  LU factorisation of v1: singular
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
lu(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 33024;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10;
    if (v9 == -1) {
	v5[v8] = v6;
	v10 = v7;
    } else {
	int64_t v11;
	if (v9 > v7 & v9 < v6) {
	    v11 = -5;
	} else {
	    bool v12 = v9 == v7;
	    int8_t *v13;
	    if (v12) {
		void *v14 = aligned_alloc(64, 33024);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v8;
	    int32_t *v17 = (int32_t *)(v15 + v16);
	    int64_t v18 = (int64_t)256;
	    double *v19 = (double *)(v15 + v18);
	    ulm_lam_copy(v19, v1, 32768);
	    int64_t v20 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v19, 64, 1, v17, 1);
	    bool v21 = (bool)(v20 != v7);
	    uint64_t v22 = (uint64_t)v21;
	    int64_t v23 = (int64_t)v22;
	    if (v23 == v7) {
		ulm_lam_copy(v3, v2, 16384);
		ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v19, 64, 1,
		                               v17, 1, v3, 32, 1);
	    }
	    if (v12) {
		free(v15);
	    }
	    v11 = v23;
	}
	v10 = v11;
    }
    return v10;
}
/* lu_t: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][32]  in    read only
 *   v3  double[64][32]  out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  LU factorisation of v1: singular
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
lu_t(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 33024;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10;
    if (v9 == -1) {
	v5[v8] = v6;
	v10 = v7;
    } else {
	int64_t v11;
	if (v9 > v7 & v9 < v6) {
	    v11 = -5;
	} else {
	    bool v12 = v9 == v7;
	    int8_t *v13;
	    if (v12) {
		void *v14 = aligned_alloc(64, 33024);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v8;
	    int32_t *v17 = (int32_t *)(v15 + v16);
	    int64_t v18 = (int64_t)256;
	    double *v19 = (double *)(v15 + v18);
	    ulm_lam_copy(v19, v1, 32768);
	    int64_t v20 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v19, 64, 1, v17, 1);
	    bool v21 = (bool)(v20 != v7);
	    uint64_t v22 = (uint64_t)v21;
	    int64_t v23 = (int64_t)v22;
	    if (v23 == v7) {
		ulm_lam_copy(v3, v2, 16384);
		ulmBLAS::getrs<std::ptrdiff_t>(true, false, 64, 32, v19, 64, 1,
		                               v17, 1, v3, 32, 1);
	    }
	    if (v12) {
		free(v15);
	    }
	    v11 = v23;
	}
	v10 = v11;
    }
    return v10;
}
/* upper: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][32]  in   read only
 *   v3  double[64][32]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
upper(double *v1, double *v2, double *v3)
{
    ulm_lam_copy(v3, v2, 16384);
    (void)ulmBLAS::trtrs<std::ptrdiff_t>(false, false, false, false, 64, 32, v1,
                                         64, 1, v3, 32, 1);
    return 0;
}
/* right: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
right(double *v1, double *v2, double *v3)
{
    ulm_lam_copy(v3, v1, 16384);
    (void)ulmBLAS::trtrs<std::ptrdiff_t>(false, false, false, false, 64, 32, v2,
                                         1, 64, v3, 1, 64);
    return 0;
}
/* right_t: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
right_t(double *v1, double *v2, double *v3)
{
    ulm_lam_copy(v3, v1, 16384);
    (void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false, 64, 32, v2,
                                         64, 1, v3, 1, 64);
    return 0;
}
/* chol_vec: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][1]   in    read only
 *   v3  double[64][1]   out   written
 *   v4  int8_t *        work  workspace (bytes)
 *   v5  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v1: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
chol_vec(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 32768;
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10;
    if (v9 == -1) {
	v5[v8] = v6;
	v10 = v7;
    } else {
	int64_t v11;
	if (v9 > v7 & v9 < v6) {
	    v11 = -5;
	} else {
	    bool v12 = v9 == v7;
	    int8_t *v13;
	    if (v12) {
		void *v14 = aligned_alloc(64, 32768);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v8;
	    double *v17 = (double *)(v15 + v16);
	    ulm_lam_copy(v17, v1, 32768);
	    int64_t v18 = ulmBLAS::potrf<std::ptrdiff_t>(true, 64, v17, 64, 1);
	    bool v19 = (bool)(v18 != v7);
	    uint64_t v20 = (uint64_t)v19;
	    int64_t v21 = (int64_t)v20;
	    if (v21 == v7) {
		ulm_lam_copy(v3, v2, 512);
		ulmBLAS::trlsv<std::ptrdiff_t>(64, false, v17, 64, 1, v3, 1);
		ulmBLAS::trusv<std::ptrdiff_t>(64, false, v17, 1, 64, v3, 1);
	    }
	    if (v12) {
		free(v15);
	    }
	    v11 = v21;
	}
	v10 = v11;
    }
    return v10;
}
/* sum: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[64][64]  in   read only
 *   v4  double[64][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
sum(double *v1, double *v2, double *v3, double *v4)
{
    double v5 = 2.00000000000000000e+00;
    double v6 = -1.00000000000000000e+00;
    ulm_lam_copy(v4, v1, 32768);
    ulmBLAS::geaxpy<std::ptrdiff_t>(64, 64, v5, v2, 64, 1, v4, 64, 1);
    ulmBLAS::geaxpy<std::ptrdiff_t>(64, 64, v6, v3, 64, 1, v4, 64, 1);
    return 0;
}
/* mixed: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in    read only
 *   v2  double[64][64]  in    read only
 *   v3  double[64][32]  in    read only
 *   v4  double[64][32]  in    read only
 *   v5  double[64][32]  out   written
 *   v6  int8_t *        work  workspace (bytes)
 *   v7  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  LU factorisation of v1: singular
 *    -7  v7: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
mixed(double *v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
      int64_t *v7)
{
    int64_t v8 = 49408;
    double v9 = 1.00000000000000000e+00;
    double v10 = 0.0e+00;
    int64_t v11 = 0;
    size_t v12 = 0;
    int64_t v13 = v7[v12];
    int64_t v14;
    if (v13 == -1) {
	v7[v12] = v8;
	v14 = v11;
    } else {
	int64_t v15;
	if (v13 > v11 & v13 < v8) {
	    v15 = -7;
	} else {
	    bool v16 = v13 == v11;
	    int8_t *v17;
	    if (v16) {
		void *v18 = aligned_alloc(64, 49408);
		v17 = (int8_t *)v18;
	    } else {
		v17 = v6;
	    }
	    int8_t *v19 = v17;
	    int64_t v20 = (int64_t)v12;
	    int32_t *v21 = (int32_t *)(v19 + v20);
	    int64_t v22 = (int64_t)256;
	    double *v23 = (double *)(v19 + v22);
	    ulm_lam_copy(v23, v1, 32768);
	    int64_t v24 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v23, 64, 1, v21, 1);
	    bool v25 = (bool)(v24 != v11);
	    uint64_t v26 = (uint64_t)v25;
	    int64_t v27 = (int64_t)v26;
	    if (v27 == v11) {
		int64_t v28 = (int64_t)33024;
		double *v29 = (double *)(v19 + v28);
		ulmBLAS::gemm<std::ptrdiff_t>(64, 32, 64, v9, false, v2, 64, 1,
		                              false, v3, 32, 1, v10, v29, 32,
		                              1);
		ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v23, 64, 1,
		                               v21, 1, v29, 32, 1);
		ulm_lam_copy(v5, v4, 16384);
		ulmBLAS::geaxpy<std::ptrdiff_t>(64, 32, v9, v29, 32, 1, v5, 32,
		                                1);
	    }
	    if (v16) {
		free(v19);
	    }
	    v15 = v27;
	}
	v14 = v15;
    }
    return v14;
}
