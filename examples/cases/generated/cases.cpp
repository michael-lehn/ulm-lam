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
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][32]  inout  may be overwritten (unspecified on exit)
 *   v3  double[64][32]  out    written
 *   v4  int8_t *        work   workspace (bytes)
 *   v5  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  LU factorisation of v1: singular
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
lu(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 256;
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
		void *v14 = aligned_alloc(64, 256);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v8;
	    int32_t *v17 = (int32_t *)(v15 + v16);
	    int64_t v18 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v1, 64, 1, v17, 1);
	    bool v19 = (bool)(v18 != v7);
	    uint64_t v20 = (uint64_t)v19;
	    int64_t v21 = (int64_t)v20;
	    if (v21 == v7) {
		ulm_lam_copy(v3, v2, 16384);
		ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v1, 64, 1,
		                               v17, 1, v3, 32, 1);
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
/* lu_keep: matrices by reference (row-major), owned by the caller
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
lu_keep(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    int64_t v7 = -1;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10 = (int64_t)v8;
    double *v11 = (double *)(v4 + v10);
    int64_t v12 = (int64_t)1;
    double *v13 = (double *)(v4 + v12);
    int64_t v14 = (int64_t)2;
    double *v15 = (double *)(v4 + v14);
    v5[v8] = v7;
    int64_t v16 = lu(v11, v13, v15, v4, v5);
    int64_t v17;
    if (v16 == v6) {
	int64_t v18 = v5[v8];
	uint64_t v19 = (uint64_t)v18;
	uint64_t v20 = (uint64_t)63;
	int64_t v21 = (int64_t)(v19 + v20);
	uint64_t v22 = (uint64_t)v21;
	uint64_t v23 = (uint64_t)-64;
	int64_t v24 = (int64_t)(v22 & v23);
	uint64_t v25 = (uint64_t)v24;
	uint64_t v26 = (uint64_t)32768;
	int64_t v27 = (int64_t)(v25 + v26);
	v5[v8] = v9;
	int64_t v28;
	if (v9 == v7) {
	    v5[v8] = v27;
	    v28 = v6;
	} else {
	    int64_t v29;
	    if (v9 > v6 & v9 < v27) {
		v29 = -5;
	    } else {
		bool v30 = v9 == v6;
		int8_t *v31;
		if (v30) {
		    ptrdiff_t v32 = (ptrdiff_t)v27;
		    size_t v33 = (size_t)v32;
		    void *v34 = aligned_alloc(64, v33);
		    v31 = (int8_t *)v34;
		} else {
		    v31 = v4;
		}
		int8_t *v35 = v31;
		int64_t v36 = (int64_t)v8;
		double *v37 = (double *)(v35 + v36);
		ulm_lam_copy(v37, v1, 32768);
		int64_t v38 = (int64_t)32768;
		v5[v8] = v18;
		int64_t v39 = lu(v37, v2, v3, v35 + v38, v5);
		if (v39 == v6) {
		    v5[v8] = v9;
		}
		if (v30) {
		    free(v35);
		}
		v29 = v39;
	    }
	    v28 = v29;
	}
	v17 = v28;
    } else {
	v17 = v16;
    }
    return v17;
}
/* lu_t: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][32]  inout  may be overwritten (unspecified on exit)
 *   v3  double[64][32]  out    written
 *   v4  int8_t *        work   workspace (bytes)
 *   v5  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  LU factorisation of v1: singular
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
lu_t(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 256;
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
		void *v14 = aligned_alloc(64, 256);
		v13 = (int8_t *)v14;
	    } else {
		v13 = v4;
	    }
	    int8_t *v15 = v13;
	    int64_t v16 = (int64_t)v8;
	    int32_t *v17 = (int32_t *)(v15 + v16);
	    int64_t v18 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v1, 64, 1, v17, 1);
	    bool v19 = (bool)(v18 != v7);
	    uint64_t v20 = (uint64_t)v19;
	    int64_t v21 = (int64_t)v20;
	    if (v21 == v7) {
		ulm_lam_copy(v3, v2, 16384);
		ulmBLAS::getrs<std::ptrdiff_t>(true, false, 64, 32, v1, 64, 1,
		                               v17, 1, v3, 32, 1);
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
/* lu_t_keep: matrices by reference (row-major), owned by the caller
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
lu_t_keep(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    int64_t v7 = -1;
    size_t v8 = 0;
    int64_t v9 = v5[v8];
    int64_t v10 = (int64_t)v8;
    double *v11 = (double *)(v4 + v10);
    int64_t v12 = (int64_t)1;
    double *v13 = (double *)(v4 + v12);
    int64_t v14 = (int64_t)2;
    double *v15 = (double *)(v4 + v14);
    v5[v8] = v7;
    int64_t v16 = lu_t(v11, v13, v15, v4, v5);
    int64_t v17;
    if (v16 == v6) {
	int64_t v18 = v5[v8];
	uint64_t v19 = (uint64_t)v18;
	uint64_t v20 = (uint64_t)63;
	int64_t v21 = (int64_t)(v19 + v20);
	uint64_t v22 = (uint64_t)v21;
	uint64_t v23 = (uint64_t)-64;
	int64_t v24 = (int64_t)(v22 & v23);
	uint64_t v25 = (uint64_t)v24;
	uint64_t v26 = (uint64_t)32768;
	int64_t v27 = (int64_t)(v25 + v26);
	v5[v8] = v9;
	int64_t v28;
	if (v9 == v7) {
	    v5[v8] = v27;
	    v28 = v6;
	} else {
	    int64_t v29;
	    if (v9 > v6 & v9 < v27) {
		v29 = -5;
	    } else {
		bool v30 = v9 == v6;
		int8_t *v31;
		if (v30) {
		    ptrdiff_t v32 = (ptrdiff_t)v27;
		    size_t v33 = (size_t)v32;
		    void *v34 = aligned_alloc(64, v33);
		    v31 = (int8_t *)v34;
		} else {
		    v31 = v4;
		}
		int8_t *v35 = v31;
		int64_t v36 = (int64_t)v8;
		double *v37 = (double *)(v35 + v36);
		ulm_lam_copy(v37, v1, 32768);
		int64_t v38 = (int64_t)32768;
		v5[v8] = v18;
		int64_t v39 = lu_t(v37, v2, v3, v35 + v38, v5);
		if (v39 == v6) {
		    v5[v8] = v9;
		}
		if (v30) {
		    free(v35);
		}
		v29 = v39;
	    }
	    v28 = v29;
	}
	v17 = v28;
    } else {
	v17 = v16;
    }
    return v17;
}
/* upper: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][32]  inout  may be overwritten (unspecified on exit)
 *   v3  double[64][32]  out    written
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
/* upper_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][32]  in   read only
 *   v3  double[64][32]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
upper_keep(double *v1, double *v2, double *v3)
{
    int64_t v4 = upper(v1, v2, v3);
    return v4;
}
/* right: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v3  double[32][64]  out    written
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
/* right_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
right_keep(double *v1, double *v2, double *v3)
{
    int64_t v4 = right(v1, v2, v3);
    return v4;
}
/* right_t: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v3  double[32][64]  out    written
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
/* right_t_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[32][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[32][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
right_t_keep(double *v1, double *v2, double *v3)
{
    int64_t v4 = right_t(v1, v2, v3);
    return v4;
}
/* chol_vec: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][1]   inout  may be overwritten (unspecified on exit)
 *   v3  double[64][1]   out    written
 * returns info:
 *    0  success
 *    1  Cholesky factorisation of v1: not positive definite
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
chol_vec(double *v1, double *v2, double *v3)
{
    int64_t v4 = 0;
    int64_t v5 = ulmBLAS::potrf<std::ptrdiff_t>(true, 64, v1, 64, 1);
    bool v6 = (bool)(v5 != v4);
    uint64_t v7 = (uint64_t)v6;
    int64_t v8 = (int64_t)v7;
    if (v8 == v4) {
	ulm_lam_copy(v3, v2, 512);
	ulmBLAS::trlsv<std::ptrdiff_t>(64, false, v1, 64, 1, v3, 1);
	ulmBLAS::trusv<std::ptrdiff_t>(64, false, v1, 1, 64, v3, 1);
    }
    return v8;
}
/* chol_vec_keep: matrices by reference (row-major), owned by the caller
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
chol_vec_keep(double *v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
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
	    int64_t v18 = chol_vec(v17, v2, v3);
	    if (v12) {
		free(v15);
	    }
	    v11 = v18;
	}
	v10 = v11;
    }
    return v10;
}
/* sum: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v3  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v4  double[64][64]  out    written
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
/* sum_keep: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  in   read only
 *   v2  double[64][64]  in   read only
 *   v3  double[64][64]  in   read only
 *   v4  double[64][64]  out  written
 * returns info:
 *    0  success
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
sum_keep(double *v1, double *v2, double *v3, double *v4)
{
    int64_t v5 = sum(v1, v2, v3, v4);
    return v5;
}
/* mixed: matrices by reference (row-major), owned by the caller
 *   v1  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v2  double[64][64]  inout  may be overwritten (unspecified on exit)
 *   v3  double[64][32]  inout  may be overwritten (unspecified on exit)
 *   v4  double[64][32]  inout  may be overwritten (unspecified on exit)
 *   v5  double[64][32]  out    written
 *   v6  int8_t *        work   workspace (bytes)
 *   v7  int64_t *       info   -1: query, 0: allocate, >0: size of work
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
    int64_t v8 = 16640;
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
		void *v18 = aligned_alloc(64, 16640);
		v17 = (int8_t *)v18;
	    } else {
		v17 = v6;
	    }
	    int8_t *v19 = v17;
	    int64_t v20 = (int64_t)v12;
	    int32_t *v21 = (int32_t *)(v19 + v20);
	    int64_t v22 =
	        ulmBLAS::getrf<std::ptrdiff_t>(64, 64, v1, 64, 1, v21, 1);
	    bool v23 = (bool)(v22 != v11);
	    uint64_t v24 = (uint64_t)v23;
	    int64_t v25 = (int64_t)v24;
	    if (v25 == v11) {
		int64_t v26 = (int64_t)256;
		double *v27 = (double *)(v19 + v26);
		ulmBLAS::gemm<std::ptrdiff_t>(64, 32, 64, v9, false, v2, 64, 1,
		                              false, v3, 32, 1, v10, v27, 32,
		                              1);
		ulmBLAS::getrs<std::ptrdiff_t>(false, false, 64, 32, v1, 64, 1,
		                               v21, 1, v27, 32, 1);
		ulm_lam_copy(v5, v4, 16384);
		ulmBLAS::geaxpy<std::ptrdiff_t>(64, 32, v9, v27, 32, 1, v5, 32,
		                                1);
	    }
	    if (v16) {
		free(v19);
	    }
	    v15 = v25;
	}
	v14 = v15;
    }
    return v14;
}
/* mixed_keep: matrices by reference (row-major), owned by the caller
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
mixed_keep(double *v1, double *v2, double *v3, double *v4, double *v5,
           int8_t *v6, int64_t *v7)
{
    int64_t v8 = 0;
    int64_t v9 = -1;
    size_t v10 = 0;
    int64_t v11 = v7[v10];
    int64_t v12 = (int64_t)v10;
    double *v13 = (double *)(v6 + v12);
    int64_t v14 = (int64_t)1;
    double *v15 = (double *)(v6 + v14);
    int64_t v16 = (int64_t)2;
    double *v17 = (double *)(v6 + v16);
    int64_t v18 = (int64_t)3;
    double *v19 = (double *)(v6 + v18);
    int64_t v20 = (int64_t)4;
    double *v21 = (double *)(v6 + v20);
    v7[v10] = v9;
    int64_t v22 = mixed(v13, v15, v17, v19, v21, v6, v7);
    int64_t v23;
    if (v22 == v8) {
	int64_t v24 = v7[v10];
	uint64_t v25 = (uint64_t)v24;
	uint64_t v26 = (uint64_t)63;
	int64_t v27 = (int64_t)(v25 + v26);
	uint64_t v28 = (uint64_t)v27;
	uint64_t v29 = (uint64_t)-64;
	int64_t v30 = (int64_t)(v28 & v29);
	uint64_t v31 = (uint64_t)v30;
	uint64_t v32 = (uint64_t)32768;
	int64_t v33 = (int64_t)(v31 + v32);
	v7[v10] = v11;
	int64_t v34;
	if (v11 == v9) {
	    v7[v10] = v33;
	    v34 = v8;
	} else {
	    int64_t v35;
	    if (v11 > v8 & v11 < v33) {
		v35 = -7;
	    } else {
		bool v36 = v11 == v8;
		int8_t *v37;
		if (v36) {
		    ptrdiff_t v38 = (ptrdiff_t)v33;
		    size_t v39 = (size_t)v38;
		    void *v40 = aligned_alloc(64, v39);
		    v37 = (int8_t *)v40;
		} else {
		    v37 = v6;
		}
		int8_t *v41 = v37;
		int64_t v42 = (int64_t)v10;
		double *v43 = (double *)(v41 + v42);
		ulm_lam_copy(v43, v1, 32768);
		int64_t v44 = (int64_t)32768;
		v7[v10] = v24;
		int64_t v45 = mixed(v43, v2, v3, v4, v5, v41 + v44, v7);
		if (v45 == v8) {
		    v7[v10] = v11;
		}
		if (v36) {
		    free(v41);
		}
		v35 = v45;
	    }
	    v34 = v35;
	}
	v23 = v34;
    } else {
	v23 = v22;
    }
    return v23;
}
