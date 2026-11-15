#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <typename T>
static void
ulm_lam_gecopy(std::ptrdiff_t m, std::ptrdiff_t n, const T *A,
               std::ptrdiff_t incRowA, std::ptrdiff_t incColA, T *B,
               std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    const std::ptrdiff_t bs = 64;
#pragma omp parallel for collapse(2) schedule(static) if (m * n > 1 << 16)
    for (std::ptrdiff_t i0 = 0; i0 < m; i0 += bs)
	for (std::ptrdiff_t j0 = 0; j0 < n; j0 += bs)
	    for (std::ptrdiff_t i = i0; i < i0 + bs && i < m; ++i)
		for (std::ptrdiff_t j = j0; j < j0 + bs && j < n; ++j)
		    B[i * incRowB + j * incColB] = A[i * incRowA + j * incColA];
}

template <typename T>
static void
ulm_lam_sycomplete(bool lower, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
                   std::ptrdiff_t incCol)
{
    // in tiles of b x b: an element-wise transposed copy would touch a new
    // cache line with every write
    const std::ptrdiff_t b = 64;
    for (std::ptrdiff_t ib = 0; ib < n; ib += b)
	for (std::ptrdiff_t jb = 0; jb <= ib; jb += b)
	    for (std::ptrdiff_t i = ib; i < n && i < ib + b; ++i)
		for (std::ptrdiff_t j = jb; j < i && j < jb + b; ++j) {
		    T &lo = A[i * incRow + j * incCol],
		      &up = A[j * incRow + i * incCol];
		    if (lower)
			up = lo;
		    else
			lo = up;
		}
}
extern "C" void
measure(size_t, size_t, double *);
extern "C" void
estimate(size_t, size_t, double *);
/* kalman: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v4   double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v5   double[v2][v1]  inout  may be overwritten (unspecified on exit)
 *   v6   double[v2][v2]  inout  may be overwritten (unspecified on exit)
 *   v7   double[v1][1]   inout  may be overwritten (unspecified on exit)
 *   v8   double[v1][v1]  inout  may be overwritten (unspecified on exit)
 *   v9   size_t          in
 *   v10  double[v1][1]   out    written
 *   v11  double[v1][v1]  out    written
 *   v12  int8_t *        work   workspace (bytes)
 *   v13  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a matrix computed from v6: not positive
 *         definite
 *    -13  v13: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
kalman(size_t v1, size_t v2, double *v3, double *v4, double *v5, double *v6,
       double *v7, double *v8, size_t v9, double *v10, double *v11, int8_t *v12,
       int64_t *v13)
{
    double v14 = -1.00000000000000000e+00;
    double v15 = 1.00000000000000000e+00;
    double v16 = 0.0e+00;
    int64_t v17 = 0;
    size_t v18 = -64;
    size_t v19 = 63;
    size_t v20 = 8;
    size_t v21 = 0;
    int64_t v22 = v13[v21];
    size_t v23 = v1 * v20;
    size_t v24 = v23 * v1 + v19 & v18;
    size_t v25 = v2 * v20;
    size_t v26 = v24 + (v25 * v1 + v19 & v18);
    size_t v27 = v26 + (v23 * v2 + v19 & v18);
    size_t v28 = v27 + (v25 + v19 & v18);
    size_t v29 = v23 + v19 & v18;
    size_t v30 = v28 + v29;
    size_t v31 = v30 + v24;
    size_t v32 = v31 + v24;
    size_t v33 = v32 + (v25 * v2 + v19 & v18);
    size_t v34 = v33 + v29;
    ptrdiff_t v35 = (ptrdiff_t)v34;
    int64_t v36 = (int64_t)v35;
    int64_t v37;
    if (v22 == -1) {
	v13[v21] = v36;
	v37 = v17;
    } else {
	int64_t v38;
	if (v22 > v17 & v22 < v36) {
	    v38 = -13;
	} else {
	    bool v39 = v22 == v17;
	    int8_t *v40;
	    if (v39) {
		void *v41 = aligned_alloc(64, v34);
		v40 = (int8_t *)v41;
	    } else {
		v40 = v12;
	    }
	    int8_t *v42 = v40;
	    int64_t v43 = (int64_t)v21;
	    double *v44 = (double *)(v42 + v43);
	    int64_t v45 = (int64_t)v24;
	    double *v46 = (double *)(v42 + v45);
	    int64_t v47 = (int64_t)v26;
	    double *v48 = (double *)(v42 + v47);
	    int64_t v49 = (int64_t)v27;
	    double *v50 = (double *)(v42 + v49);
	    int64_t v51;
	    v51 = v17;
	    for (size_t i52 = v21; i52 < v9; i52 += 1) {
		int64_t v53 = v51;
		int64_t v54;
		if (v53 == v17) {
		    int64_t v55 = (int64_t)v28;
		    double *v56 = (double *)(v42 + v55);
		    size_t v57 = 8 * v1;
		    ulm_lam_copy(v56, v7, v57);
		    ulmBLAS::gemv<std::ptrdiff_t>(v1, v1, v15, v3, v1, 1, v7, 1,
		                                  v16, v56, 1);
		    ulmBLAS::gemm<std::ptrdiff_t>(v1, v1, v1, v15, false, v3,
		                                  v1, 1, false, v8, v1, 1, v16,
		                                  v44, v1, 1);
		    int64_t v58 = (int64_t)v30;
		    double *v59 = (double *)(v42 + v58);
		    size_t v60 = (8 * v1) * v1;
		    ulm_lam_copy(v59, v4, v60);
		    ulmBLAS::gemmtr<std::ptrdiff_t>(true, v1, v1, v15, false,
		                                    v44, v1, 1, false, v3, 1,
		                                    v1, v15, v59, v1, 1);
		    int64_t v61 = (int64_t)v31;
		    double *v62 = (double *)(v42 + v61);
		    size_t v63 = (8 * v1) * v1;
		    ulm_lam_copy(v62, v59, v63);
		    ulm_lam_sycomplete(true, v1, v62, v1, 1);
		    ulmBLAS::sylmm<std::ptrdiff_t>(v1, v2, v15, v59, v1, 1, v5,
		                                   1, v1, v16, v46, 1, v1);
		    int64_t v64 = (int64_t)v32;
		    double *v65 = (double *)(v42 + v64);
		    size_t v66 = (8 * v2) * v2;
		    ulm_lam_copy(v65, v6, v66);
		    ulmBLAS::gemmtr<std::ptrdiff_t>(true, v2, v1, v15, false,
		                                    v46, v1, 1, false, v5, 1,
		                                    v1, v15, v65, v2, 1);
		    int64_t v67 =
		        ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v65, v2, 1);
		    bool v68 = (bool)(v67 != v17);
		    uint64_t v69 = (uint64_t)v68;
		    int64_t v70 = (int64_t)v69;
		    if (v70 == v17) {
			ulm_lam_gecopy(v1, v2, v46, 1, v1, v48, v2, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
			                                     false, v2, v1, v65,
			                                     v2, 1, v48, 1, v2);
			measure(v2, i52, v50);
			ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v14, v5, v1, 1,
			                              v56, 1, v15, v50, 1);
			ulmBLAS::trlsv<std::ptrdiff_t>(v2, false, v65, v2, 1,
			                               v50, 1);
			ulmBLAS::gemv<std::ptrdiff_t>(v1, v2, v15, v48, v2, 1,
			                              v50, 1, v15, v56, 1);
			ulmBLAS::sylrk<std::ptrdiff_t>(v1, v2, v14, v48, v2, 1,
			                               v15, v62, v1, 1);
			ulm_lam_sycomplete(true, v1, v62, v1, 1);
			int64_t v71 = (int64_t)v33;
			double *v72 = (double *)(v42 + v71);
			size_t v73 = 8 * v1;
			ulm_lam_copy(v72, v56, v73);
			estimate(v1, i52, v72);
			size_t v74 = 8 * v1;
			ulm_lam_copy(v7, v56, v74);
			size_t v75 = (8 * v1) * v1;
			ulm_lam_copy(v8, v62, v75);
		    }
		    v54 = v70;
		} else {
		    v54 = v53;
		}
		v51 = v54;
	    }
	    int64_t v76 = v51;
	    if (v76 == v17) {
		size_t v77 = 8 * v1;
		ulm_lam_copy(v10, v7, v77);
		size_t v78 = (8 * v1) * v1;
		ulm_lam_copy(v11, v8, v78);
	    }
	    if (v39) {
		free(v42);
	    }
	    v38 = v76;
	}
	v37 = v38;
    }
    return v37;
}
/* kalman_keep: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   double[v1][v1]  in    read only
 *   v4   double[v1][v1]  in    read only
 *   v5   double[v2][v1]  in    read only
 *   v6   double[v2][v2]  in    read only
 *   v7   double[v1][1]   in    read only
 *   v8   double[v1][v1]  in    read only
 *   v9   size_t          in
 *   v10  double[v1][1]   out   written
 *   v11  double[v1][v1]  out   written
 *   v12  int8_t *        work  workspace (bytes)
 *   v13  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a matrix computed from v6: not positive
 *         definite
 *    -13  v13: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
kalman_keep(size_t v1, size_t v2, double *v3, double *v4, double *v5,
            double *v6, double *v7, double *v8, size_t v9, double *v10,
            double *v11, int8_t *v12, int64_t *v13)
{
    int64_t v14 = 0;
    int64_t v15 = -1;
    size_t v16 = -64;
    size_t v17 = 63;
    size_t v18 = 0;
    int64_t v19 = v13[v18];
    size_t v20 = v1 * 8;
    size_t v21 = v20 + v17 & v16;
    size_t v22 = v21 + (v20 * v1 + v17 & v16);
    ptrdiff_t v23 = (ptrdiff_t)v22;
    int64_t v24 = (int64_t)v23;
    int64_t v25 = (int64_t)2;
    double *v26 = (double *)(v12 + v25);
    int64_t v27 = (int64_t)3;
    double *v28 = (double *)(v12 + v27);
    int64_t v29 = (int64_t)4;
    double *v30 = (double *)(v12 + v29);
    int64_t v31 = (int64_t)5;
    double *v32 = (double *)(v12 + v31);
    int64_t v33 = (int64_t)6;
    double *v34 = (double *)(v12 + v33);
    int64_t v35 = (int64_t)7;
    double *v36 = (double *)(v12 + v35);
    int64_t v37 = (int64_t)9;
    double *v38 = (double *)(v12 + v37);
    int64_t v39 = (int64_t)10;
    double *v40 = (double *)(v12 + v39);
    v13[v18] = v15;
    int64_t v41 =
        kalman(v1, v2, v26, v28, v30, v32, v34, v36, v9, v38, v40, v12, v13);
    int64_t v42;
    if (v41 == v14) {
	int64_t v43 = v13[v18];
	uint64_t v44 = (uint64_t)v43;
	uint64_t v45 = (uint64_t)63;
	int64_t v46 = (int64_t)(v44 + v45);
	uint64_t v47 = (uint64_t)v46;
	uint64_t v48 = (uint64_t)-64;
	int64_t v49 = (int64_t)(v47 & v48);
	uint64_t v50 = (uint64_t)v24;
	uint64_t v51 = (uint64_t)v49;
	int64_t v52 = (int64_t)(v50 + v51);
	v13[v18] = v19;
	int64_t v53;
	if (v19 == v15) {
	    v13[v18] = v52;
	    v53 = v14;
	} else {
	    int64_t v54;
	    if (v19 > v14 & v19 < v52) {
		v54 = -13;
	    } else {
		bool v55 = v19 == v14;
		int8_t *v56;
		if (v55) {
		    ptrdiff_t v57 = (ptrdiff_t)v52;
		    size_t v58 = (size_t)v57;
		    void *v59 = aligned_alloc(64, v58);
		    v56 = (int8_t *)v59;
		} else {
		    v56 = v12;
		}
		int8_t *v60 = v56;
		int64_t v61 = (int64_t)v18;
		double *v62 = (double *)(v60 + v61);
		size_t v63 = 8 * v1;
		ulm_lam_copy(v62, v7, v63);
		int64_t v64 = (int64_t)v21;
		double *v65 = (double *)(v60 + v64);
		size_t v66 = (8 * v1) * v1;
		ulm_lam_copy(v65, v8, v66);
		int64_t v67 = (int64_t)v22;
		v13[v18] = v43;
		int64_t v68 = kalman(v1, v2, v3, v4, v5, v6, v62, v65, v9, v10,
		                     v11, v60 + v67, v13);
		if (v68 == v14) {
		    v13[v18] = v19;
		}
		if (v55) {
		    free(v60);
		}
		v54 = v68;
	    }
	    v53 = v54;
	}
	v42 = v53;
    } else {
	v42 = v41;
    }
    return v42;
}
