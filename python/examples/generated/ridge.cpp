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
ulm_lam_adddiag(std::ptrdiff_t n, T alpha, T *A, std::ptrdiff_t incRow,
                std::ptrdiff_t incCol)
{
    for (std::ptrdiff_t i = 0; i < n; ++i)
	A[i * (incRow + incCol)] += alpha;
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

/* ridge: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  size_t          in
 *   v4  double[v1][v2]  inout  may be overwritten (unspecified on exit)
 *   v5  double[v1][v3]  inout  may be overwritten (unspecified on exit)
 *   v6  double          in
 *   v7  double[v2][v3]  out    written
 *   v8  int8_t *        work   workspace (bytes)
 *   v9  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of a computed matrix: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -9  v9: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
ridge(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double v6,
      double *v7, int8_t *v8, int64_t *v9)
{
    double v10 = 1.00000000000000000e+00;
    double v11 = 0.0e+00;
    int64_t v12 = 0;
    size_t v13 = 2;
    size_t v14 = -64;
    size_t v15 = 63;
    size_t v16 = 8;
    size_t v17 = 0;
    int64_t v18 = v9[v17];
    size_t v19 = v1 * v16;
    size_t v20 = v19 * v1 + v15 & v14;
    size_t v21 = v2 * v16;
    size_t v22 = v21 * v3 + v15 & v14;
    size_t v23 = v20 + v22;
    size_t v24 = v23 + (v19 * v2 + v15 & v14);
    size_t v25 = v21 * v2 + v15 & v14;
    size_t v26 = v24 + v25;
    size_t v27 = v23 + v20;
    size_t v28 = v23 + v22;
    size_t v29 = v2 * v1;
    size_t v30 = (v29 * v3) * v13;
    size_t v31 = v1 * v1;
    size_t v32 = v31 * v3;
    size_t v33 = (((v30 + v30) + v32) + v32) + v30;
    size_t v34 = v29 * v1;
    size_t v35 = (v34 + v31 * v1) + v31 * v1;
    size_t v36 = (v35 + v34 * v13) + v30;
    bool v37 = v33 <= v36;
    size_t v38 = v31 * v2 + v29 * v2;
    size_t v39 = v2 * v2;
    size_t v40 = v39 * v3;
    size_t v41 = (v38 + v30) + v40 * v13;
    bool v42 = v41 <= v33 & v41 <= v36;
    size_t v43 = (v35 + v32 * v13) + v30;
    bool v44 = (v43 <= v41 & v43 <= v33) & v43 <= v36;
    size_t v45 = v39 * v1;
    bool v46 =
        (((v38 + v45 * v13) + v30 <= v43 & (v38 + v45 * v13) + v30 <= v41) &
	 (v38 + v45 * v13) + v30 <= v33) &
        (v38 + v45 * v13) + v30 <= v36;
    bool v47 = (v30 + v40) + v40 <= ((v45 + v29) + v45) + v30;
    bool v48 = v1 < v2;
    size_t v49 =
        v48 ? (v46 ? v26 + (v21 * v1 + v15 & v14)
	           : (v44 ? v27 + (v19 * v3 + v15 & v14)
	                  : (v42 ? v26 + v22
	                         : (v37 ? v28 + (v19 * v3 + v15 & v14)
	                                : v27 + (v21 * v1 + v15 & v14)))))
	    : (v47 ? v25 : v25 + (v21 * v1 + v15 & v14));
    ptrdiff_t v50 = (ptrdiff_t)v49;
    int64_t v51 = (int64_t)v50;
    int64_t v52;
    if (v18 == -1) {
	v9[v17] = v51;
	v52 = v12;
    } else {
	int64_t v53;
	if (v18 > v12 & v18 < v51) {
	    v53 = -9;
	} else {
	    bool v54 = v18 == v12;
	    int8_t *v55;
	    if (v54) {
		void *v56 = aligned_alloc(64, v49);
		v55 = (int8_t *)v56;
	    } else {
		v55 = v8;
	    }
	    int8_t *v57 = v55;
	    int64_t v58;
	    if (v48) {
		int64_t v59 = (int64_t)v17;
		double *v60 = (double *)(v57 + v59);
		ulmBLAS::sylrk<std::ptrdiff_t>(v1, v2, v10, v4, v2, 1, v11, v60,
		                               v1, 1);
		ulm_lam_adddiag(v1, v6, v60, v1, 1);
		int64_t v61 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v60, v1, 1);
		bool v62 = (bool)(v61 != v12);
		uint64_t v63 = (uint64_t)v62;
		int64_t v64 = (int64_t)v63;
		if (v64 == v12) {
		    double v65 = v10 / v6;
		    ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v65, false, v4, 1,
		                                  v2, false, v5, v3, 1, v11, v7,
		                                  v3, 1);
		    int64_t v66 = (int64_t)v20;
		    double *v67 = (double *)(v57 + v66);
		    if (v46) {
			int64_t v68 = (int64_t)v23;
			double *v69 = (double *)(v57 + v68);
			size_t v70 = (8 * v1) * v2;
			ulm_lam_copy(v69, v4, v70);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
			                                     false, v1, v2, v60,
			                                     v1, 1, v69, v2, 1);
			int64_t v71 = (int64_t)v24;
			double *v72 = (double *)(v57 + v71);
			ulmBLAS::gemm<std::ptrdiff_t>(v2, v2, v1, v10, false,
			                              v69, 1, v2, false, v69,
			                              v2, 1, v11, v72, v2, 1);
			int64_t v73 = (int64_t)v26;
			double *v74 = (double *)(v57 + v73);
			ulmBLAS::gemm<std::ptrdiff_t>(v2, v1, v2, v10, false,
			                              v72, v2, 1, false, v4, 1,
			                              v2, v11, v74, v1, 1);
			ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false,
			                              v74, v1, 1, false, v5, v3,
			                              1, v11, v67, v3, 1);
		    } else {
			if (v44) {
			    int64_t v75 = (int64_t)v23;
			    double *v76 = (double *)(v57 + v75);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v1, v1, v2, v10, false, v4, v2, 1, false, v4, 1,
			        v2, v11, v76, v1, 1);
			    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
			        true, false, false, false, v1, v1, v60, v1, 1,
			        v76, v1, 1);
			    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
			        false, false, false, false, v1, v1, v60, 1, v1,
			        v76, v1, 1);
			    int64_t v77 = (int64_t)v27;
			    double *v78 = (double *)(v57 + v77);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v1, v3, v1, v10, false, v76, v1, 1, false, v5,
			        v3, 1, v11, v78, v3, 1);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v2, v3, v1, v10, false, v4, 1, v2, false, v78,
			        v3, 1, v11, v67, v3, 1);
			} else {
			    if (v42) {
				int64_t v79 = (int64_t)v23;
				double *v80 = (double *)(v57 + v79);
				size_t v81 = (8 * v1) * v2;
				ulm_lam_copy(v80, v4, v81);
				(void)ulmBLAS::trtrs<std::ptrdiff_t>(
				    true, false, false, false, v1, v2, v60, v1,
				    1, v80, v2, 1);
				int64_t v82 = (int64_t)v24;
				double *v83 = (double *)(v57 + v82);
				ulmBLAS::gemm<std::ptrdiff_t>(
				    v2, v2, v1, v10, false, v80, 1, v2, false,
				    v80, v2, 1, v11, v83, v2, 1);
				int64_t v84 = (int64_t)v26;
				double *v85 = (double *)(v57 + v84);
				ulmBLAS::gemm<std::ptrdiff_t>(
				    v2, v3, v1, v10, false, v4, 1, v2, false,
				    v5, v3, 1, v11, v85, v3, 1);
				ulmBLAS::gemm<std::ptrdiff_t>(
				    v2, v3, v2, v10, false, v83, v2, 1, false,
				    v85, v3, 1, v11, v67, v3, 1);
			    } else {
				if (v37) {
				    int64_t v86 = (int64_t)v23;
				    double *v87 = (double *)(v57 + v86);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v2, v3, v1, v10, false, v4, 1, v2,
				        false, v5, v3, 1, v11, v87, v3, 1);
				    int64_t v88 = (int64_t)v28;
				    double *v89 = (double *)(v57 + v88);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v1, v3, v2, v10, false, v4, v2, 1,
				        false, v87, v3, 1, v11, v89, v3, 1);
				    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
				        true, false, false, false, v1, v3, v60,
				        v1, 1, v89, v3, 1);
				    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
				        false, false, false, false, v1, v3, v60,
				        1, v1, v89, v3, 1);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v2, v3, v1, v10, false, v4, 1, v2,
				        false, v89, v3, 1, v11, v67, v3, 1);
				} else {
				    int64_t v90 = (int64_t)v23;
				    double *v91 = (double *)(v57 + v90);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v1, v1, v2, v10, false, v4, v2, 1,
				        false, v4, 1, v2, v11, v91, v1, 1);
				    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
				        true, false, false, false, v1, v1, v60,
				        v1, 1, v91, v1, 1);
				    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
				        false, false, false, false, v1, v1, v60,
				        1, v1, v91, v1, 1);
				    int64_t v92 = (int64_t)v27;
				    double *v93 = (double *)(v57 + v92);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v2, v1, v1, v10, false, v4, 1, v2,
				        false, v91, v1, 1, v11, v93, v1, 1);
				    ulmBLAS::gemm<std::ptrdiff_t>(
				        v2, v3, v1, v10, false, v93, v1, 1,
				        false, v5, v3, 1, v11, v67, v3, 1);
				}
			    }
			}
		    }
		    double v94 = v65 * -1.00000000000000000e+00;
		    ulmBLAS::gescal<std::ptrdiff_t>(v2, v3, v94, v67, v3, 1);
		    ulmBLAS::geaxpy<std::ptrdiff_t>(v2, v3, v10, v67, v3, 1, v7,
		                                    v3, 1);
		}
		v58 = v64;
	    } else {
		int64_t v95 = (int64_t)v17;
		double *v96 = (double *)(v57 + v95);
		ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v10, v4, 1, v2, v11, v96,
		                               v2, 1);
		ulm_lam_adddiag(v2, v6, v96, v2, 1);
		int64_t v97 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v96, v2, 1);
		int64_t v98 = v97 != v12 ? 2 : v12;
		if (v98 == v12) {
		    if (v47) {
			ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false,
			                              v4, 1, v2, false, v5, v3,
			                              1, v11, v7, v3, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
			                                     false, v2, v3, v96,
			                                     v2, 1, v7, v3, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, v2, v3, v96, 1, v2, v7,
			    v3, 1);
		    } else {
			int64_t v99 = (int64_t)v25;
			double *v100 = (double *)(v57 + v99);
			ulm_lam_gecopy(v2, v1, v4, 1, v2, v100, v1, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    true, false, false, false, v2, v1, v96, v2, 1, v100,
			    v1, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, v2, v1, v96, 1, v2,
			    v100, v1, 1);
			ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false,
			                              v100, v1, 1, false, v5,
			                              v3, 1, v11, v7, v3, 1);
		    }
		}
		v58 = v98;
	    }
	    int64_t v101 = v58;
	    if (v54) {
		free(v57);
	    }
	    v53 = v101;
	}
	v52 = v53;
    }
    return v52;
}
/* ridge_keep: matrices by reference (row-major), owned by the caller
 *   v1  size_t          in
 *   v2  size_t          in
 *   v3  size_t          in
 *   v4  double[v1][v2]  in    read only
 *   v5  double[v1][v3]  in    read only
 *   v6  double          in
 *   v7  double[v2][v3]  out   written
 *   v8  int8_t *        work  workspace (bytes)
 *   v9  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of a computed matrix: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -9  v9: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
ridge_keep(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double v6,
           double *v7, int8_t *v8, int64_t *v9)
{
    int64_t v10 = 0;
    int64_t v11 = -1;
    size_t v12 = 0;
    int64_t v13 = v9[v12];
    int64_t v14 = (int64_t)3;
    double *v15 = (double *)(v8 + v14);
    int64_t v16 = (int64_t)4;
    double *v17 = (double *)(v8 + v16);
    int64_t v18 = (int64_t)6;
    double *v19 = (double *)(v8 + v18);
    v9[v12] = v11;
    int64_t v20 = ridge(v1, v2, v3, v15, v17, v6, v19, v8, v9);
    int64_t v21;
    if (v20 == v10) {
	int64_t v22 = v9[v12];
	uint64_t v23 = (uint64_t)v22;
	uint64_t v24 = (uint64_t)63;
	int64_t v25 = (int64_t)(v23 + v24);
	uint64_t v26 = (uint64_t)v25;
	uint64_t v27 = (uint64_t)-64;
	int64_t v28 = (int64_t)(v26 & v27);
	v9[v12] = v13;
	int64_t v29;
	if (v13 == v11) {
	    v9[v12] = v28;
	    v29 = v10;
	} else {
	    int64_t v30;
	    if (v13 > v10 & v13 < v28) {
		v30 = -9;
	    } else {
		bool v31 = v13 == v10;
		int8_t *v32;
		if (v31) {
		    ptrdiff_t v33 = (ptrdiff_t)v28;
		    size_t v34 = (size_t)v33;
		    void *v35 = aligned_alloc(64, v34);
		    v32 = (int8_t *)v35;
		} else {
		    v32 = v8;
		}
		int8_t *v36 = v32;
		int64_t v37 = (int64_t)v12;
		v9[v12] = v22;
		int64_t v38 = ridge(v1, v2, v3, v4, v5, v6, v7, v36 + v37, v9);
		if (v38 == v10) {
		    v9[v12] = v13;
		}
		if (v31) {
		    free(v36);
		}
		v30 = v38;
	    }
	    v29 = v30;
	}
	v21 = v29;
    } else {
	v21 = v20;
    }
    return v21;
}
