#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/trtrs.h>
#include <ulmblas/ulmblas.h>
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
 *   v4  double[v1][v2]  in    read only
 *   v5  double[v1][v3]  in    read only
 *   v6  double          in
 *   v7  double[v2][v3]  out   written
 *   v8  int8_t *        work  workspace (bytes)
 *   v9  int64_t *       info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of a computed matrix: not positive definite
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
    size_t v13 = -64;
    size_t v14 = 63;
    size_t v15 = 0;
    int64_t v16 = v9[v15];
    size_t v17 = v2 * 8;
    size_t v18 = v17 * v2 + v14 & v13;
    bool v19 =
        (((v2 * v1) * v3) * 2 + (v2 * v2) * v3) + (v2 * v2) * v3 <=
        (((v2 * v2) * v1 + v2 * v1) + (v2 * v2) * v1) + ((v2 * v1) * v3) * 2;
    size_t v20 = v19 ? v18 : v18 + (v17 * v1 + v14 & v13);
    ptrdiff_t v21 = (ptrdiff_t)v20;
    int64_t v22 = (int64_t)v21;
    int64_t v23;
    if (v16 == -1) {
	v9[v15] = v22;
	v23 = v12;
    } else {
	int64_t v24;
	if (v16 > v12 & v16 < v22) {
	    v24 = -9;
	} else {
	    bool v25 = v16 == v12;
	    int8_t *v26;
	    if (v25) {
		void *v27 = aligned_alloc(64, v20);
		v26 = (int8_t *)v27;
	    } else {
		v26 = v8;
	    }
	    int8_t *v28 = v26;
	    int64_t v29 = (int64_t)v15;
	    double *v30 = (double *)(v28 + v29);
	    ulmBLAS::sylrk<std::ptrdiff_t>(v2, v1, v10, v4, 1, v2, v11, v30, v2,
	                                   1);
	    ulm_lam_adddiag(v2, v6, v30, v2, 1);
	    int64_t v31 = ulmBLAS::potrf<std::ptrdiff_t>(true, v2, v30, v2, 1);
	    bool v32 = (bool)(v31 != v12);
	    uint64_t v33 = (uint64_t)v32;
	    int64_t v34 = (int64_t)v33;
	    if (v34 == v12) {
		if (v19) {
		    ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false, v4, 1,
		                                  v2, false, v5, v3, 1, v11, v7,
		                                  v3, 1);
		    (void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
		                                         false, v2, v3, v30, v2,
		                                         1, v7, v3, 1);
		    (void)ulmBLAS::trtrs<std::ptrdiff_t>(false, false, false,
		                                         false, v2, v3, v30, 1,
		                                         v2, v7, v3, 1);
		} else {
		    int64_t v35 = (int64_t)v18;
		    double *v36 = (double *)(v28 + v35);
		    ulm_lam_gecopy(v2, v1, v4, 1, v2, v36, v1, 1);
		    (void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false,
		                                         false, v2, v1, v30, v2,
		                                         1, v36, v1, 1);
		    (void)ulmBLAS::trtrs<std::ptrdiff_t>(false, false, false,
		                                         false, v2, v1, v30, 1,
		                                         v2, v36, v1, 1);
		    ulmBLAS::gemm<std::ptrdiff_t>(v2, v3, v1, v10, false, v36,
		                                  v1, 1, false, v5, v3, 1, v11,
		                                  v7, v3, 1);
		}
	    }
	    if (v25) {
		free(v28);
	    }
	    v24 = v34;
	}
	v23 = v24;
    }
    return v23;
}
