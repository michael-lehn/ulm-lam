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

/* bar: matrices by reference (row-major), owned by the caller
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
bar(size_t v1, double *v2, double *v3, double *v4, int8_t *v5, int64_t *v6)
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
	    int64_t v22 = ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v20, v1, 1);
	    bool v23 = (bool)(v22 != v7);
	    uint64_t v24 = (uint64_t)v23;
	    int64_t v25 = (int64_t)v24;
	    if (v25 == v7) {
		size_t v26 = 8 * v1;
		ulm_lam_copy(v4, v3, v26);
		ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v20, v1, 1, v4, 1);
		ulmBLAS::trusv<std::ptrdiff_t>(v1, false, v20, 1, v1, v4, 1);
	    }
	    if (v15) {
		free(v18);
	    }
	    v14 = v25;
	}
	v13 = v14;
    }
    return v13;
}
/* foo: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  double[v1][v1]  in    read only
 *   v3  double[v1][v1]  in    read only
 *   v4  double[v1][1]   in    read only
 *   v5  double[v1][1]   out   written
 *   v6  int8_t *        work  workspace (bytes)
 *   v7  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  call of bar returned info != 0
 *     2  call of bar returned info != 0
 *    -7  v7: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
foo(size_t v1, double *v2, double *v3, double *v4, double *v5, int8_t *v6,
    int64_t *v7)
{
    int64_t v8 = 0;
    int64_t v9 = -64;
    int64_t v10 = 63;
    int64_t v11 = -1;
    size_t v12 = 0;
    int64_t v13 = v7[v12];
    size_t v14 = v1 * 8 + 63 & -64;
    ptrdiff_t v15 = (ptrdiff_t)v14;
    int64_t v16 = (int64_t)v15;
    int64_t v17 = (int64_t)1;
    double *v18 = (double *)(v6 + v17);
    int64_t v19 = (int64_t)2;
    double *v20 = (double *)(v6 + v19);
    int64_t v21 = (int64_t)3;
    double *v22 = (double *)(v6 + v21);
    v7[v12] = v11;
    int64_t v23 = bar(v1, v18, v20, v22, v6, v7);
    int64_t v24;
    if (v23 == v8) {
	int64_t v25 = v7[v12];
	uint64_t v26 = (uint64_t)v25;
	uint64_t v27 = (uint64_t)v10;
	int64_t v28 = (int64_t)(v26 + v27);
	uint64_t v29 = (uint64_t)v28;
	uint64_t v30 = (uint64_t)v9;
	int64_t v31 = (int64_t)(v29 & v30);
	uint64_t v32 = (uint64_t)v16;
	uint64_t v33 = (uint64_t)v31;
	int64_t v34 = (int64_t)(v32 + v33);
	v7[v12] = v11;
	int64_t v35 = bar(v1, v18, v20, v22, v6, v7);
	int64_t v36;
	if (v35 == v8) {
	    int64_t v37 = v7[v12];
	    uint64_t v38 = (uint64_t)v37;
	    uint64_t v39 = (uint64_t)v10;
	    int64_t v40 = (int64_t)(v38 + v39);
	    uint64_t v41 = (uint64_t)v40;
	    uint64_t v42 = (uint64_t)v9;
	    int64_t v43 = (int64_t)(v41 & v42);
	    uint64_t v44 = (uint64_t)v34;
	    uint64_t v45 = (uint64_t)v43;
	    int64_t v46 = (int64_t)(v44 + v45);
	    v7[v12] = v13;
	    int64_t v47;
	    if (v13 == v11) {
		v7[v12] = v46;
		v47 = v8;
	    } else {
		int64_t v48;
		if (v13 > v8 & v13 < v46) {
		    v48 = -7;
		} else {
		    bool v49 = v13 == v8;
		    int8_t *v50;
		    if (v49) {
			ptrdiff_t v51 = (ptrdiff_t)v46;
			size_t v52 = (size_t)v51;
			void *v53 = aligned_alloc(64, v52);
			v50 = (int8_t *)v53;
		    } else {
			v50 = v6;
		    }
		    int8_t *v54 = v50;
		    int64_t v55 = (int64_t)v12;
		    double *v56 = (double *)(v54 + v55);
		    int64_t v57 = (int64_t)v14;
		    v7[v12] = v25;
		    bool v58 =
		        (bool)(bar(v1, v3, v4, v56, v54 + v57, v7) != v8);
		    uint64_t v59 = (uint64_t)v58;
		    int64_t v60 = (int64_t)v59;
		    int64_t v61;
		    if (v60 == v8) {
			ptrdiff_t v62 = (ptrdiff_t)v34;
			size_t v63 = (size_t)v62;
			int64_t v64 = (int64_t)v63;
			v7[v12] = v37;
			int64_t v65 =
			    bar(v1, v2, v56, v5, v54 + v64, v7) != v8 ? 2 : v8;
			if (v65 == v8) {
			    v7[v12] = v13;
			}
			v61 = v65;
		    } else {
			v61 = v60;
		    }
		    int64_t v66 = v61;
		    if (v49) {
			free(v54);
		    }
		    v48 = v66;
		}
		v47 = v48;
	    }
	    v36 = v47;
	} else {
	    v36 = v35;
	}
	v24 = v36;
    } else {
	v24 = v23;
    }
    return v24;
}
