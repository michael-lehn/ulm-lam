#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
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

/* bar_: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v3  double[v1][1]   inout  updated in place
 * returns info:
 *    0  success
 *    1  Cholesky factorisation of v2: not positive definite
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
bar_(size_t v1, double *v2, double *v3)
{
    int64_t v4 = 0;
    int64_t v5 = ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v2, v1, 1);
    bool v6 = (bool)(v5 != v4);
    uint64_t v7 = (uint64_t)v6;
    int64_t v8 = (int64_t)v7;
    if (v8 == v4) {
	ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v2, v1, 1, v3, 1);
	ulmBLAS::trusv<std::ptrdiff_t>(v1, false, v2, 1, v1, v3, 1);
    }
    return v8;
}
/* bar__keep: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in     read only
 *   v3  double[v1][1]   inout  updated in place
 *   v4  int8_t *        work   workspace (bytes)
 *   v5  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
bar__keep(size_t v1, double *v2, double *v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 0;
    size_t v7 = 0;
    int64_t v8 = v5[v7];
    size_t v9 = (v1 * 8) * v1 + 63 & -64;
    ptrdiff_t v10 = (ptrdiff_t)v9;
    int64_t v11 = (int64_t)v10;
    int64_t v12;
    if (v8 == -1) {
	v5[v7] = v11;
	v12 = v6;
    } else {
	int64_t v13;
	if (v8 > v6 & v8 < v11) {
	    v13 = -5;
	} else {
	    bool v14 = v8 == v6;
	    int8_t *v15;
	    if (v14) {
		void *v16 = aligned_alloc(64, v9);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v4;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 = (int64_t)v7;
	    double *v19 = (double *)(v17 + v18);
	    size_t v20 = (8 * v1) * v1;
	    ulm_lam_copy(v19, v2, v20);
	    int64_t v21 = bar_(v1, v19, v3);
	    if (v14) {
		free(v17);
	    }
	    v13 = v21;
	}
	v12 = v13;
    }
    return v12;
}
/* foo_: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v3  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v4  double[v1][1]   inout  updated in place
 * returns info:
 *    0  success
 *    1  call of bar_ returned info != 0
 *    2  call of bar_ returned info != 0
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo_(size_t v1, double *v2, double *v3, double *v4)
{
    int64_t v5 = 0;
    bool v6 = (bool)(bar_(v1, v3, v4) != v5);
    uint64_t v7 = (uint64_t)v6;
    int64_t v8 = (int64_t)v7;
    int64_t v9;
    if (v8 == v5) {
	int64_t v10 = bar_(v1, v2, v4) != v5 ? 2 : v5;
	v9 = v10;
    } else {
	v9 = v8;
    }
    return v9;
}
/* foo__keep: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in     read only
 *   v3  double[v1][v1]  in     read only
 *   v4  double[v1][1]   inout  updated in place
 *   v5  int8_t *        work   workspace (bytes)
 *   v6  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of bar_ returned info != 0
 *     2  call of bar_ returned info != 0
 *    -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo__keep(size_t v1, double *v2, double *v3, double *v4, int8_t *v5,
          int64_t *v6)
{
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v6[v8];
    size_t v10 = (v1 * 8) * v1 + 63 & -64;
    size_t v11 = v10 + v10;
    ptrdiff_t v12 = (ptrdiff_t)v11;
    int64_t v13 = (int64_t)v12;
    int64_t v14;
    if (v9 == -1) {
	v6[v8] = v13;
	v14 = v7;
    } else {
	int64_t v15;
	if (v9 > v7 & v9 < v13) {
	    v15 = -6;
	} else {
	    bool v16 = v9 == v7;
	    int8_t *v17;
	    if (v16) {
		void *v18 = aligned_alloc(64, v11);
		v17 = (int8_t *)v18;
	    } else {
		v17 = v5;
	    }
	    int8_t *v19 = v17;
	    int64_t v20 = (int64_t)v8;
	    double *v21 = (double *)(v19 + v20);
	    size_t v22 = (8 * v1) * v1;
	    ulm_lam_copy(v21, v2, v22);
	    int64_t v23 = (int64_t)v10;
	    double *v24 = (double *)(v19 + v23);
	    size_t v25 = (8 * v1) * v1;
	    ulm_lam_copy(v24, v3, v25);
	    int64_t v26 = foo_(v1, v21, v24, v4);
	    if (v16) {
		free(v19);
	    }
	    v15 = v26;
	}
	v14 = v15;
    }
    return v14;
}
/* foo_matlab: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in    read only
 *   v3  double[v1][v1]  in    read only
 *   v4  double[v1][1]   in    read only
 *   v5  double[v1][1]   out   written
 *   v6  int8_t *        work  workspace (bytes)
 *   v7  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of foo_ returned info != 0
 *    -7  v7: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo_matlab(size_t v1, double *v2, double *v3, double *v4, double *v5,
           int8_t *v6, int64_t *v7)
{
    int64_t v8 = 0;
    size_t v9 = 0;
    int64_t v10 = v7[v9];
    size_t v11 = (v1 * 8) * v1 + 63 & -64;
    size_t v12 = v11 + v11;
    ptrdiff_t v13 = (ptrdiff_t)v12;
    int64_t v14 = (int64_t)v13;
    int64_t v15;
    if (v10 == -1) {
	v7[v9] = v14;
	v15 = v8;
    } else {
	int64_t v16;
	if (v10 > v8 & v10 < v14) {
	    v16 = -7;
	} else {
	    bool v17 = v10 == v8;
	    int8_t *v18;
	    if (v17) {
		void *v19 = aligned_alloc(64, v12);
		v18 = (int8_t *)v19;
	    } else {
		v18 = v6;
	    }
	    int8_t *v20 = v18;
	    size_t v21 = 8 * v1;
	    ulm_lam_copy(v5, v4, v21);
	    int64_t v22 = (int64_t)v9;
	    double *v23 = (double *)(v20 + v22);
	    size_t v24 = (8 * v1) * v1;
	    ulm_lam_copy(v23, v2, v24);
	    int64_t v25 = (int64_t)v11;
	    double *v26 = (double *)(v20 + v25);
	    size_t v27 = (8 * v1) * v1;
	    ulm_lam_copy(v26, v3, v27);
	    bool v28 = (bool)(foo_(v1, v23, v26, v5) != v8);
	    uint64_t v29 = (uint64_t)v28;
	    int64_t v30 = (int64_t)v29;
	    if (v17) {
		free(v20);
	    }
	    v16 = v30;
	}
	v15 = v16;
    }
    return v15;
}
