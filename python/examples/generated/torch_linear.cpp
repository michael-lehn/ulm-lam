#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" void
batch_x(size_t, size_t, size_t, double *);
extern "C" void
batch_y(size_t, size_t, size_t, double *);
/* train: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   size_t          in
 *   v4   double[v1][v2]  in    read only
 *   v5   double[v2][1]   in    read only
 *   v6   double[v3][1]   in    read only
 *   v7   double          in
 *   v8   size_t          in
 *   v9   double[v1][v2]  out   written
 *   v10  double[v2][1]   out   written
 *   v11  int8_t *        work  workspace (bytes)
 *   v12  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *    -12  v12: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
train(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double *v6,
      double v7, size_t v8, double *v9, double *v10, int8_t *v11, int64_t *v12)
{
    double v13 = -1.00000000000000000e+00;
    double v14 = 1.00000000000000000e+00;
    int64_t v15 = 0;
    size_t v16 = -64;
    size_t v17 = 63;
    size_t v18 = 8;
    size_t v19 = 0;
    int64_t v20 = v12[v19];
    size_t v21 = (v1 * v18) * v2 + v17 & v16;
    size_t v22 = v21 + (v2 * v18 + v17 & v16);
    size_t v23 = v3 * v18;
    size_t v24 = v22 + (v23 * v1 + v17 & v16);
    size_t v25 = v24 + (v23 * v2 + v17 & v16);
    ptrdiff_t v26 = (ptrdiff_t)v25;
    int64_t v27 = (int64_t)v26;
    int64_t v28;
    if (v20 == -1) {
	v12[v19] = v27;
	v28 = v15;
    } else {
	bool v29 = v20 > v15 & v20 < v27;
	if (v29) {
	} else {
	    bool v30 = v20 == v15;
	    int8_t *v31;
	    if (v30) {
		void *v32 = aligned_alloc(64, v25);
		v31 = (int8_t *)v32;
	    } else {
		v31 = v11;
	    }
	    int8_t *v33 = v31;
	    double v34 = v7 * v13;
	    int64_t v35 = (int64_t)v19;
	    double *v36 = (double *)(v33 + v35);
	    size_t v37 = (8 * v1) * v2;
	    ulm_lam_copy(v36, v4, v37);
	    int64_t v38 = (int64_t)v21;
	    double *v39 = (double *)(v33 + v38);
	    size_t v40 = 8 * v2;
	    ulm_lam_copy(v39, v5, v40);
	    int64_t v41 = (int64_t)v22;
	    double *v42 = (double *)(v33 + v41);
	    int64_t v43 = (int64_t)v24;
	    double *v44 = (double *)(v33 + v43);
	    for (size_t i45 = v19; i45 < v8; i45 += 1) {
		batch_x(v3, v1, i45, v42);
		batch_y(v3, v2, i45, v44);
		ulmBLAS::gemm<std::ptrdiff_t>(v3, v2, v1, v14, false, v42, v1,
		                              1, false, v36, v2, 1, v13, v44,
		                              v2, 1);
		ulmBLAS::gemm<std::ptrdiff_t>(v3, v2, 1, v14, false, v6, 1, 1,
		                              false, v39, 1, 1, v14, v44, v2,
		                              1);
		ulmBLAS::gemm<std::ptrdiff_t>(v1, v2, v3, v34, false, v42, 1,
		                              v1, false, v44, v2, 1, v14, v36,
		                              v2, 1);
		ulmBLAS::gemv<std::ptrdiff_t>(v2, v3, v34, v44, 1, v2, v6, 1,
		                              v14, v39, 1);
	    }
	    size_t v46 = (8 * v1) * v2;
	    ulm_lam_copy(v9, v36, v46);
	    size_t v47 = 8 * v2;
	    ulm_lam_copy(v10, v39, v47);
	    if (v30) {
		free(v33);
	    }
	}
	v28 = v29 ? -12 : v15;
    }
    return v28;
}
