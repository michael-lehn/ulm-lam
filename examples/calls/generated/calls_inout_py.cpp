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
    int64_t v6 = ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v2, v1, 1);
    bool v7 = (bool)(v6 != v5);
    uint64_t v8 = (uint64_t)v7;
    int64_t v9 = (int64_t)v8;
    if (v9 == v5) {
	size_t v10 = 8 * v1;
	ulm_lam_copy(v4, v3, v10);
	ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v2, v1, 1, v4, 1);
	ulmBLAS::trusv<std::ptrdiff_t>(v1, false, v2, 1, v1, v4, 1);
    }
    return v9;
}
/* bar2_keep: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in    read only
 *   v3  double[v1][1]   in    read only
 *   v4  double[v1][1]   out   written
 *   v5  int8_t *        work  workspace (bytes)
 *   v6  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *    -6  v6: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
bar2_keep(size_t v1, double *v2, double *v3, double *v4, int8_t *v5,
          int64_t *v6)
{
    int64_t v7 = 0;
    size_t v8 = 0;
    int64_t v9 = v6[v8];
    size_t v10 = (v1 * 8) * v1 + 63 & -64;
    ptrdiff_t v11 = (ptrdiff_t)v10;
    int64_t v12 = (int64_t)v11;
    int64_t v13;
    if (v9 == -1) {
	v6[v8] = v12;
	v13 = v7;
    } else {
	int64_t v14;
	if (v9 > v7 & v9 < v12) {
	    v14 = -6;
	} else {
	    bool v15 = v9 == v7;
	    int8_t *v16;
	    if (v15) {
		void *v17 = aligned_alloc(64, v10);
		v16 = (int8_t *)v17;
	    } else {
		v16 = v5;
	    }
	    int8_t *v18 = v16;
	    int64_t v19 = (int64_t)v8;
	    double *v20 = (double *)(v18 + v19);
	    size_t v21 = (8 * v1) * v1;
	    ulm_lam_copy(v20, v2, v21);
	    int64_t v22 = bar2(v1, v20, v3, v4);
	    if (v15) {
		free(v18);
	    }
	    v14 = v22;
	}
	v13 = v14;
    }
    return v13;
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
    size_t v17 = v16 + v15;
    ptrdiff_t v18 = (ptrdiff_t)v17;
    int64_t v19 = (int64_t)v18;
    int64_t v20;
    if (v12 == -1) {
	v7[v11] = v19;
	v20 = v8;
    } else {
	int64_t v21;
	if (v12 > v8 & v12 < v19) {
	    v21 = -7;
	} else {
	    bool v22 = v12 == v8;
	    int8_t *v23;
	    if (v22) {
		void *v24 = aligned_alloc(64, v17);
		v23 = (int8_t *)v24;
	    } else {
		v23 = v6;
	    }
	    int8_t *v25 = v23;
	    int64_t v26 = (int64_t)v11;
	    double *v27 = (double *)(v25 + v26);
	    int64_t v28 = (int64_t)v14;
	    double *v29 = (double *)(v25 + v28);
	    size_t v30 = (8 * v1) * v1;
	    ulm_lam_copy(v29, v3, v30);
	    bool v31 = (bool)(bar2(v1, v29, v4, v27) != v8);
	    uint64_t v32 = (uint64_t)v31;
	    int64_t v33 = (int64_t)v32;
	    int64_t v34;
	    if (v33 == v8) {
		int64_t v35 = (int64_t)v16;
		double *v36 = (double *)(v25 + v35);
		size_t v37 = (8 * v1) * v1;
		ulm_lam_copy(v36, v2, v37);
		int64_t v38 = bar2(v1, v36, v27, v5) != v8 ? 2 : v8;
		v34 = v38;
	    } else {
		v34 = v33;
	    }
	    int64_t v39 = v34;
	    if (v22) {
		free(v25);
	    }
	    v21 = v39;
	}
	v20 = v21;
    }
    return v20;
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
    ptrdiff_t v12 = (ptrdiff_t)v11;
    int64_t v13 = (int64_t)v12;
    int64_t v14;
    if (v10 == -1) {
	v7[v9] = v13;
	v14 = v8;
    } else {
	int64_t v15;
	if (v10 > v8 & v10 < v13) {
	    v15 = -7;
	} else {
	    bool v16 = v10 == v8;
	    int8_t *v17;
	    if (v16) {
		void *v18 = aligned_alloc(64, v11);
		v17 = (int8_t *)v18;
	    } else {
		v17 = v6;
	    }
	    int8_t *v19 = v17;
	    int64_t v20 = (int64_t)v9;
	    double *v21 = (double *)(v19 + v20);
	    bool v22 = (bool)(bar2(v1, v3, v4, v21) != v8);
	    uint64_t v23 = (uint64_t)v22;
	    int64_t v24 = (int64_t)v23;
	    int64_t v25;
	    if (v24 == v8) {
		int64_t v26 = bar2(v1, v2, v21, v5) != v8 ? 2 : v8;
		v25 = v26;
	    } else {
		v25 = v24;
	    }
	    int64_t v27 = v25;
	    if (v16) {
		free(v19);
	    }
	    v15 = v27;
	}
	v14 = v15;
    }
    return v14;
}
/* foo3_keep: matrices by reference (row-major), owned by the caller
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
foo3_keep(size_t v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
          int64_t *v7)
{
    int64_t v8 = 0;
    int64_t v9 = -1;
    size_t v10 = 0;
    int64_t v11 = v7[v10];
    size_t v12 = (v1 * 8) * v1 + 63 & -64;
    size_t v13 = v12 + v12;
    ptrdiff_t v14 = (ptrdiff_t)v13;
    int64_t v15 = (int64_t)v14;
    int64_t v16 = (int64_t)1;
    double *v17 = (double *)(v6 + v16);
    int64_t v18 = (int64_t)2;
    double *v19 = (double *)(v6 + v18);
    int64_t v20 = (int64_t)3;
    double *v21 = (double *)(v6 + v20);
    int64_t v22 = (int64_t)4;
    double *v23 = (double *)(v6 + v22);
    v7[v10] = v9;
    int64_t v24 = foo3(v1, v17, v19, v21, v23, v6, v7);
    int64_t v25;
    if (v24 == v8) {
	int64_t v26 = v7[v10];
	uint64_t v27 = (uint64_t)v26;
	uint64_t v28 = (uint64_t)63;
	int64_t v29 = (int64_t)(v27 + v28);
	uint64_t v30 = (uint64_t)v29;
	uint64_t v31 = (uint64_t)-64;
	int64_t v32 = (int64_t)(v30 & v31);
	uint64_t v33 = (uint64_t)v15;
	uint64_t v34 = (uint64_t)v32;
	int64_t v35 = (int64_t)(v33 + v34);
	v7[v10] = v11;
	int64_t v36;
	if (v11 == v9) {
	    v7[v10] = v35;
	    v36 = v8;
	} else {
	    int64_t v37;
	    if (v11 > v8 & v11 < v35) {
		v37 = -7;
	    } else {
		bool v38 = v11 == v8;
		int8_t *v39;
		if (v38) {
		    ptrdiff_t v40 = (ptrdiff_t)v35;
		    size_t v41 = (size_t)v40;
		    void *v42 = aligned_alloc(64, v41);
		    v39 = (int8_t *)v42;
		} else {
		    v39 = v6;
		}
		int8_t *v43 = v39;
		int64_t v44 = (int64_t)v10;
		double *v45 = (double *)(v43 + v44);
		size_t v46 = (8 * v1) * v1;
		ulm_lam_copy(v45, v2, v46);
		int64_t v47 = (int64_t)v12;
		double *v48 = (double *)(v43 + v47);
		size_t v49 = (8 * v1) * v1;
		ulm_lam_copy(v48, v3, v49);
		int64_t v50 = (int64_t)v13;
		v7[v10] = v26;
		int64_t v51 = foo3(v1, v45, v48, v4, v5, v43 + v50, v7);
		if (v51 == v8) {
		    v7[v10] = v11;
		}
		if (v38) {
		    free(v43);
		}
		v37 = v51;
	    }
	    v36 = v37;
	}
	v25 = v36;
    } else {
	v25 = v24;
    }
    return v25;
}
