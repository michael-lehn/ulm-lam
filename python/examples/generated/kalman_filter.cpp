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

extern "C" void
measure(size_t, size_t, double *);
extern "C" void
estimate(size_t, size_t, double *);
/* kalman: matrices by reference (row-major), owned by the caller
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
    size_t v24 = v23 + v19 & v18;
    size_t v25 = v23 * v1 + v19 & v18;
    size_t v26 = v24 + v25;
    size_t v27 = v26 + v25;
    size_t v28 = v2 * v20;
    size_t v29 = v27 + (v28 * v1 + v19 & v18);
    size_t v30 = v29 + (v23 * v2 + v19 & v18);
    size_t v31 = v30 + (v28 + v19 & v18);
    size_t v32 = v31 + v24;
    size_t v33 = v32 + v25;
    size_t v34 = v33 + (v28 * v2 + v19 & v18);
    size_t v35 = v34 + v24;
    ptrdiff_t v36 = (ptrdiff_t)v35;
    int64_t v37 = (int64_t)v36;
    int64_t v38;
    if (v22 == -1) {
	v13[v21] = v37;
	v38 = v17;
    } else {
	int64_t v39;
	if (v22 > v17 & v22 < v37) {
	    v39 = -13;
	} else {
	    bool v40 = v22 == v17;
	    int8_t *v41;
	    if (v40) {
		void *v42 = aligned_alloc(64, v35);
		v41 = (int8_t *)v42;
	    } else {
		v41 = v12;
	    }
	    int8_t *v43 = v41;
	    int64_t v44 = (int64_t)v21;
	    double *v45 = (double *)(v43 + v44);
	    size_t v46 = 8 * v1;
	    ulm_lam_copy(v45, v7, v46);
	    int64_t v47 = (int64_t)v24;
	    double *v48 = (double *)(v43 + v47);
	    size_t v49 = (8 * v1) * v1;
	    ulm_lam_copy(v48, v8, v49);
	    int64_t v50 = (int64_t)v26;
	    double *v51 = (double *)(v43 + v50);
	    int64_t v52 = (int64_t)v27;
	    double *v53 = (double *)(v43 + v52);
	    int64_t v54 = (int64_t)v29;
	    double *v55 = (double *)(v43 + v54);
	    int64_t v56 = (int64_t)v30;
	    double *v57 = (double *)(v43 + v56);
	    int64_t v58;
	    v58 = v17;
	    for (size_t i59 = v21; i59 < v9; i59 += 1) {
		int64_t v60 = v58;
		int64_t v61;
		if (v60 == v17) {
		    int64_t v62 = (int64_t)v31;
		    double *v63 = (double *)(v43 + v62);
		    size_t v64 = 8 * v1;
		    ulm_lam_copy(v63, v45, v64);
		    ulmBLAS::gemv<std::ptrdiff_t>(v1, v1, v15, v3, v1, 1, v45,
		                                  1, v16, v63, 1);
		    ulmBLAS::gemm<std::ptrdiff_t>(v1, v1, v1, v15, false, v3,
		                                  v1, 1, false, v48, v1, 1, v16,
		                                  v51, v1, 1);
		    int64_t v65 = (int64_t)v32;
		    double *v66 = (double *)(v43 + v65);
		    size_t v67 = (8 * v1) * v1;
		    ulm_lam_copy(v66, v4, v67);
		    ulmBLAS::gemm<std::ptrdiff_t>(v1, v1, v1, v15, false, v51,
		                                  v1, 1, false, v3, 1, v1, v15,
		                                  v66, v1, 1);
		    ulmBLAS::gemm<std::ptrdiff_t>(v2, v1, v1, v15, false, v5,
		                                  v1, 1, false, v66, v1, 1, v16,
		                                  v53, v1, 1);
		    int64_t v68 = (int64_t)v33;
		    double *v69 = (double *)(v43 + v68);
		    size_t v70 = (8 * v2) * v2;
		    ulm_lam_copy(v69, v6, v70);
		    ulmBLAS::gemm<std::ptrdiff_t>(v2, v2, v1, v15, false, v53,
		                                  v1, 1, false, v5, 1, v1, v15,
		                                  v69, v2, 1);
		    int64_t v71 =
		        ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v69, v2, 1);
		    bool v72 = (bool)(v71 != v17);
		    uint64_t v73 = (uint64_t)v72;
		    int64_t v74 = (int64_t)v73;
		    if (v74 == v17) {
			ulm_lam_gecopy(v1, v2, v53, 1, v1, v55, v2, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
			                                     false, v2, v1, v69,
			                                     v2, 1, v55, 1, v2);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, v2, v1, v69, 1, v2, v55,
			    1, v2);
			measure(v2, i59, v57);
			ulmBLAS::gemv<std::ptrdiff_t>(v2, v1, v14, v5, v1, 1,
			                              v63, 1, v15, v57, 1);
			ulmBLAS::gemv<std::ptrdiff_t>(v1, v2, v15, v55, v2, 1,
			                              v57, 1, v15, v63, 1);
			ulmBLAS::gemm<std::ptrdiff_t>(v1, v1, v2, v14, false,
			                              v55, v2, 1, false, v53,
			                              v1, 1, v15, v66, v1, 1);
			int64_t v75 = (int64_t)v34;
			double *v76 = (double *)(v43 + v75);
			size_t v77 = 8 * v1;
			ulm_lam_copy(v76, v63, v77);
			estimate(v1, i59, v76);
			size_t v78 = 8 * v1;
			ulm_lam_copy(v45, v63, v78);
			size_t v79 = (8 * v1) * v1;
			ulm_lam_copy(v48, v66, v79);
		    }
		    v61 = v74;
		} else {
		    v61 = v60;
		}
		v58 = v61;
	    }
	    int64_t v80 = v58;
	    if (v80 == v17) {
		size_t v81 = 8 * v1;
		ulm_lam_copy(v10, v45, v81);
		size_t v82 = (8 * v1) * v1;
		ulm_lam_copy(v11, v48, v82);
	    }
	    if (v40) {
		free(v43);
	    }
	    v39 = v80;
	}
	v38 = v39;
    }
    return v38;
}
