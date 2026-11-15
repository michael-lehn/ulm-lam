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
/* blr: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   size_t          in
 *   v4   double[v1][v2]  inout  may be overwritten (unspecified on exit)
 *   v5   double[v1][1]   inout  may be overwritten (unspecified on exit)
 *   v6   double[v3][v2]  inout  may be overwritten (unspecified on exit)
 *   v7   double          in
 *   v8   double          in
 *   v9   double[v3][1]   out    written
 *   v10  double[v3][v3]  out    written
 *   v11  int8_t *        work   workspace (bytes)
 *   v12  int64_t *       info   -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a computed matrix: not positive definite
 *      2  Cholesky factorisation of a computed matrix: not positive definite
 *    -12  v12: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
blr(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double *v6,
    double v7, double v8, double *v9, double *v10, int8_t *v11, int64_t *v12)
{
    double v13 = 1.00000000000000000e+00;
    double v14 = 0.0e+00;
    int64_t v15 = 0;
    size_t v16 = 2;
    size_t v17 = -64;
    size_t v18 = 63;
    size_t v19 = 8;
    size_t v20 = 0;
    int64_t v21 = v12[v20];
    size_t v22 = v1 * v19;
    size_t v23 = v22 * v1 + v18 & v17;
    size_t v24 = v2 * v19;
    size_t v25 = v24 + v18 & v17;
    size_t v26 = v23 + v25;
    size_t v27 = v26 + (v22 + v18 & v17);
    size_t v28 = v27 + v25;
    size_t v29 = v3 * v19;
    size_t v30 = v28 + (v29 * v3 + v18 & v17);
    size_t v31 = v24 * v2 + v18 & v17;
    size_t v32 = v30 + v31;
    size_t v33 = v1 * v2;
    size_t v34 = (v33 * v3) * v16;
    size_t v35 = v1 * v1;
    size_t v36 = (v3 * v1) * v3;
    size_t v37 = (v34 + v35 * v3) + v36;
    size_t v38 = v35 * v2;
    size_t v39 =
        ((v38 + v33 * v2) + ((v3 * v2) * v2) * v16) + ((v3 * v2) * v3) * v16;
    bool v40 = v37 <= v39;
    bool v41 = (v38 + v34) + v36 <= v37 & (v38 + v34) + v36 <= v39;
    size_t v42 = v31 + v25;
    bool v43 = v1 < v2;
    size_t v44 = v43 ? (v41 ? v30 + (v22 * v3 + v18 & v17)
                            : (v40 ? v30 + (v22 * v3 + v18 & v17)
                                   : v32 + (v29 * v2 + v18 & v17)))
                     : v42 + (v24 * v3 + v18 & v17);
    ptrdiff_t v45 = (ptrdiff_t)v44;
    int64_t v46 = (int64_t)v45;
    int64_t v47;
    if (v21 == -1) {
	v12[v20] = v46;
	v47 = v15;
    } else {
	int64_t v48;
	if (v21 > v15 & v21 < v46) {
	    v48 = -12;
	} else {
	    bool v49 = v21 == v15;
	    int8_t *v50;
	    if (v49) {
		void *v51 = aligned_alloc(64, v44);
		v50 = (int8_t *)v51;
	    } else {
		v50 = v11;
	    }
	    int8_t *v52 = v50;
	    double v53 = v13 / v8;
	    int64_t v54;
	    if (v43) {
		int64_t v55 = (int64_t)v20;
		double *v56 = (double *)(v52 + v55);
		ulmBLAS::sylrk<std::ptrdiff_t>(v1, v2, v8, v4, v2, 1, v14, v56,
		                               v1, 1);
		ulm_lam_adddiag(v1, v7, v56, v1, 1);
		int64_t v57 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, v1, v56, v1, 1);
		bool v58 = (bool)(v57 != v15);
		uint64_t v59 = (uint64_t)v58;
		int64_t v60 = (int64_t)v59;
		if (v60 == v15) {
		    double v61 = v13 / v7;
		    double v62 = -v8;
		    int64_t v63 = (int64_t)v23;
		    double *v64 = (double *)(v52 + v63);
		    ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v13, v4, 1, v2, v5, 1,
		                                  v14, v64, 1);
		    double v65 = v8 * v61;
		    ulmBLAS::gemv<std::ptrdiff_t>(v3, v2, v65, v6, v2, 1, v64,
		                                  1, v14, v9, 1);
		    int64_t v66 = (int64_t)v26;
		    double *v67 = (double *)(v52 + v66);
		    ulmBLAS::gemv<std::ptrdiff_t>(v1, v2, v13, v4, v2, 1, v64,
		                                  1, v14, v67, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(v1, false, v56, v1, 1, v67,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(v1, false, v56, 1, v1, v67,
		                                   1);
		    int64_t v68 = (int64_t)v27;
		    double *v69 = (double *)(v52 + v68);
		    ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v13, v4, 1, v2, v67,
		                                  1, v14, v69, 1);
		    double v70 = (v62 * v8) * v61;
		    ulmBLAS::gemv<std::ptrdiff_t>(v3, v2, v70, v6, v2, 1, v69,
		                                  1, v13, v9, 1);
		    ulmBLAS::sylrk<std::ptrdiff_t>(v3, v2, v61, v6, v2, 1, v14,
		                                   v10, v3, 1);
		    int64_t v71 = (int64_t)v28;
		    double *v72 = (double *)(v52 + v71);
		    if (v41) {
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
			                                     false, v1, v2, v56,
			                                     v1, 1, v4, v2, 1);
			int64_t v73 = (int64_t)v30;
			double *v74 = (double *)(v52 + v73);
			ulmBLAS::gemm<std::ptrdiff_t>(v1, v3, v2, v13, false,
			                              v4, v2, 1, false, v6, 1,
			                              v2, v14, v74, v3, 1);
			ulmBLAS::gemm<std::ptrdiff_t>(v3, v3, v1, v13, false,
			                              v74, 1, v3, false, v74,
			                              v3, 1, v14, v72, v3, 1);
		    } else {
			if (v40) {
			    int64_t v75 = (int64_t)v30;
			    double *v76 = (double *)(v52 + v75);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v1, v3, v2, v13, false, v4, v2, 1, false, v6, 1,
			        v2, v14, v76, v3, 1);
			    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
			        true, false, false, false, v1, v3, v56, v1, 1,
			        v76, v3, 1);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v3, v3, v1, v13, false, v76, 1, v3, false, v76,
			        v3, 1, v14, v72, v3, 1);
			} else {
			    (void)ulmBLAS::trtrs<std::ptrdiff_t>(
			        true, false, false, false, v1, v2, v56, v1, 1,
			        v4, v2, 1);
			    int64_t v77 = (int64_t)v30;
			    double *v78 = (double *)(v52 + v77);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v2, v2, v1, v13, false, v4, 1, v2, false, v4,
			        v2, 1, v14, v78, v2, 1);
			    int64_t v79 = (int64_t)v32;
			    double *v80 = (double *)(v52 + v79);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v3, v2, v2, v13, false, v6, v2, 1, false, v78,
			        v2, 1, v14, v80, v2, 1);
			    ulmBLAS::gemm<std::ptrdiff_t>(
			        v3, v3, v2, v13, false, v80, v2, 1, false, v6,
			        1, v2, v14, v72, v3, 1);
			}
		    }
		    double v81 = v62 * v61;
		    ulmBLAS::gescal<std::ptrdiff_t>(v3, v3, v81, v72, v3, 1);
		    ulm_lam_sycomplete(true, v3, v10, v3, 1);
		    ulmBLAS::geaxpy<std::ptrdiff_t>(v3, v3, v13, v72, v3, 1,
		                                    v10, v3, 1);
		    ulm_lam_adddiag(v3, v53, v10, v3, 1);
		}
		v54 = v60;
	    } else {
		int64_t v82 = (int64_t)v20;
		double *v83 = (double *)(v52 + v82);
		ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v8, v4, 1, v2, v14, v83,
		                               v2, 1);
		ulm_lam_adddiag(v2, v7, v83, v2, 1);
		int64_t v84 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v83, v2, 1);
		int64_t v85 = v84 != v15 ? 2 : v15;
		if (v85 == v15) {
		    int64_t v86 = (int64_t)v31;
		    double *v87 = (double *)(v52 + v86);
		    ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v13, v4, 1, v2, v5, 1,
		                                  v14, v87, 1);
		    ulmBLAS::trlsv<std::ptrdiff_t>(v2, false, v83, v2, 1, v87,
		                                   1);
		    ulmBLAS::trusv<std::ptrdiff_t>(v2, false, v83, 1, v2, v87,
		                                   1);
		    ulmBLAS::gemv<std::ptrdiff_t>(v3, v2, v8, v6, v2, 1, v87, 1,
		                                  v14, v9, 1);
		    int64_t v88 = (int64_t)v42;
		    double *v89 = (double *)(v52 + v88);
		    ulm_lam_gecopy(v2, v3, v6, 1, v2, v89, v3, 1);
		    (void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
		                                         false, v2, v3, v83, v2,
		                                         1, v89, v3, 1);
		    ulmBLAS::sylrk<std::ptrdiff_t>(v3, v2, v13, v89, 1, v3, v14,
		                                   v10, v3, 1);
		    ulm_lam_adddiag(v3, v53, v10, v3, 1);
		    ulm_lam_sycomplete(true, v3, v10, v3, 1);
		}
		v54 = v85;
	    }
	    int64_t v90 = v54;
	    if (v49) {
		free(v52);
	    }
	    v48 = v90;
	}
	v47 = v48;
    }
    return v47;
}
/* blr_keep: matrices by reference (row-major), owned by the caller
 *   v1   size_t          in
 *   v2   size_t          in
 *   v3   size_t          in
 *   v4   double[v1][v2]  in    read only
 *   v5   double[v1][1]   in    read only
 *   v6   double[v3][v2]  in    read only
 *   v7   double          in
 *   v8   double          in
 *   v9   double[v3][1]   out   written
 *   v10  double[v3][v3]  out   written
 *   v11  int8_t *        work  workspace (bytes)
 *   v12  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *      0  success
 *      1  Cholesky factorisation of a computed matrix: not positive definite
 *      2  Cholesky factorisation of a computed matrix: not positive definite
 *    -12  v12: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
blr_keep(size_t v1, size_t v2, size_t v3, double *v4, double *v5, double *v6,
         double v7, double v8, double *v9, double *v10, int8_t *v11,
         int64_t *v12)
{
    int64_t v13 = 0;
    int64_t v14 = -1;
    size_t v15 = 8;
    size_t v16 = 0;
    int64_t v17 = v12[v16];
    size_t v18 = (v1 * v15) * v2 + 63 & -64;
    ptrdiff_t v19 = (ptrdiff_t)v18;
    int64_t v20 = (int64_t)v19;
    int64_t v21 = (int64_t)3;
    double *v22 = (double *)(v11 + v21);
    int64_t v23 = (int64_t)4;
    double *v24 = (double *)(v11 + v23);
    int64_t v25 = (int64_t)5;
    double *v26 = (double *)(v11 + v25);
    int64_t v27 = (int64_t)v15;
    double *v28 = (double *)(v11 + v27);
    int64_t v29 = (int64_t)9;
    double *v30 = (double *)(v11 + v29);
    v12[v16] = v14;
    int64_t v31 = blr(v1, v2, v3, v22, v24, v26, v7, v8, v28, v30, v11, v12);
    int64_t v32;
    if (v31 == v13) {
	int64_t v33 = v12[v16];
	uint64_t v34 = (uint64_t)v33;
	uint64_t v35 = (uint64_t)63;
	int64_t v36 = (int64_t)(v34 + v35);
	uint64_t v37 = (uint64_t)v36;
	uint64_t v38 = (uint64_t)-64;
	int64_t v39 = (int64_t)(v37 & v38);
	uint64_t v40 = (uint64_t)v20;
	uint64_t v41 = (uint64_t)v39;
	int64_t v42 = (int64_t)(v40 + v41);
	v12[v16] = v17;
	int64_t v43;
	if (v17 == v14) {
	    v12[v16] = v42;
	    v43 = v13;
	} else {
	    int64_t v44;
	    if (v17 > v13 & v17 < v42) {
		v44 = -12;
	    } else {
		bool v45 = v17 == v13;
		int8_t *v46;
		if (v45) {
		    ptrdiff_t v47 = (ptrdiff_t)v42;
		    size_t v48 = (size_t)v47;
		    void *v49 = aligned_alloc(64, v48);
		    v46 = (int8_t *)v49;
		} else {
		    v46 = v11;
		}
		int8_t *v50 = v46;
		int64_t v51 = (int64_t)v16;
		double *v52 = (double *)(v50 + v51);
		size_t v53 = (8 * v1) * v2;
		ulm_lam_copy(v52, v4, v53);
		int64_t v54 = (int64_t)v18;
		v12[v16] = v33;
		int64_t v55 = blr(v1, v2, v3, v52, v5, v6, v7, v8, v9, v10,
		                  v50 + v54, v12);
		if (v55 == v13) {
		    v12[v16] = v17;
		}
		if (v45) {
		    free(v50);
		}
		v44 = v55;
	    }
	    v43 = v44;
	}
	v32 = v43;
    } else {
	v32 = v31;
    }
    return v32;
}
