#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulm_lam/cblas_backend.h>
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

/* bar2: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v3  double[v1][1]   in     read only
 *   v4  double[v1][1]   out    written
 * returns info:
 *    0  success
 *    1  Cholesky factorisation of v2: not positive definite
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
bar2(size_t v1, double *v2, double *v3, double *v4)
{
    int64_t v5 = 0;
    int64_t v6 = ulm_lam_cblas::potrf<std::ptrdiff_t>(true, v1, v2, v1, 1);
    bool v7 = (bool)(v6 != v5);
    uint64_t v8 = (uint64_t)v7;
    int64_t v9 = (int64_t)v8;
    if (v9 == v5) {
	size_t v10 = 8 * v1;
	ulm_lam_copy(v4, v3, v10);
	ulm_lam_cblas::trlsv<std::ptrdiff_t>(v1, false, v2, v1, 1, v4, 1);
	ulm_lam_cblas::trusv<std::ptrdiff_t>(v1, false, v2, 1, v1, v4, 1);
    }
    return v9;
}
/* foo2: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in    read only
 *   v3  double[v1][v1]  in    read only
 *   v4  double[v1][1]   in    read only
 *   v5  double[v1][1]   out   written
 *   v6  int8_t *        work  workspace (bytes)
 *   v7  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of bar2 returned info != 0
 *     2  call of bar2 returned info != 0
 *    -7  v7: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo2(size_t v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
     int64_t *v7)
{
    int64_t v8 = 0;
    size_t v9 = -64;
    size_t v10 = 63;
    size_t v11 = 0;
    int64_t v12 = v7[v11];
    size_t v13 = v1 * 8;
    size_t v14 = v13 + v10 & v9;
    size_t v15 = v13 * v1 + v10 & v9;
    size_t v16 = v14 + v15;
    size_t v17 = v16 + v14;
    size_t v18 = v17 + v15;
    ptrdiff_t v19 = (ptrdiff_t)v18;
    int64_t v20 = (int64_t)v19;
    int64_t v21;
    if (v12 == -1) {
	v7[v11] = v20;
	v21 = v8;
    } else {
	int64_t v22;
	if (v12 > v8 & v12 < v20) {
	    v22 = -7;
	} else {
	    bool v23 = v12 == v8;
	    int8_t *v24;
	    if (v23) {
		void *v25 = aligned_alloc(64, v18);
		v24 = (int8_t *)v25;
	    } else {
		v24 = v6;
	    }
	    int8_t *v26 = v24;
	    int64_t v27 = (int64_t)v11;
	    double *v28 = (double *)(v26 + v27);
	    int64_t v29 = (int64_t)v14;
	    double *v30 = (double *)(v26 + v29);
	    size_t v31 = (8 * v1) * v1;
	    ulm_lam_copy(v30, v3, v31);
	    bool v32 = (bool)(bar2(v1, v30, v4, v28) != v8);
	    uint64_t v33 = (uint64_t)v32;
	    int64_t v34 = (int64_t)v33;
	    int64_t v35;
	    if (v34 == v8) {
		int64_t v36 = (int64_t)v16;
		double *v37 = (double *)(v26 + v36);
		int64_t v38 = (int64_t)v17;
		double *v39 = (double *)(v26 + v38);
		size_t v40 = (8 * v1) * v1;
		ulm_lam_copy(v39, v2, v40);
		int64_t v41 = bar2(v1, v39, v28, v37) != v8 ? 2 : v8;
		if (v41 == v8) {
		    size_t v42 = 8 * v1;
		    ulm_lam_copy(v5, v37, v42);
		}
		v35 = v41;
	    } else {
		v35 = v34;
	    }
	    int64_t v43 = v35;
	    if (v23) {
		free(v26);
	    }
	    v22 = v43;
	}
	v21 = v22;
    }
    return v21;
}
/* foo3: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v3  double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v4  double[v1][1]   in     read only
 *   v5  double[v1][1]   out    written
 *   v6  int8_t *        work   workspace (bytes)
 *   v7  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of bar2 returned info != 0
 *     2  call of bar2 returned info != 0
 *    -7  v7: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo3(size_t v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
     int64_t *v7)
{
    int64_t v8 = 0;
    size_t v9 = 0;
    int64_t v10 = v7[v9];
    size_t v11 = v1 * 8 + 63 & -64;
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
	    int64_t v21 = (int64_t)v9;
	    double *v22 = (double *)(v20 + v21);
	    bool v23 = (bool)(bar2(v1, v3, v4, v22) != v8);
	    uint64_t v24 = (uint64_t)v23;
	    int64_t v25 = (int64_t)v24;
	    int64_t v26;
	    if (v25 == v8) {
		int64_t v27 = (int64_t)v11;
		double *v28 = (double *)(v20 + v27);
		int64_t v29 = bar2(v1, v2, v22, v28) != v8 ? 2 : v8;
		if (v29 == v8) {
		    size_t v30 = 8 * v1;
		    ulm_lam_copy(v5, v28, v30);
		}
		v26 = v29;
	    } else {
		v26 = v25;
	    }
	    int64_t v31 = v26;
	    if (v17) {
		free(v20);
	    }
	    v16 = v31;
	}
	v15 = v16;
    }
    return v15;
}
