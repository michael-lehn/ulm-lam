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
observe(size_t, double *);
extern "C" void
record(size_t, double *);
/* estimate: matrices by reference (row-major), owned by the caller
 *   v1  double[2000][200]   in    read only
 *   v2  double[2000][2000]  in    read only
 *   v3  size_t              in
 *   v4  int8_t *            work  workspace (bytes)
 *   v5  int64_t *           info  -1: query, 0: allocate, >0: size of work
 * returns info:
 *     0  success
 *     1  Cholesky factorisation of v2: not positive definite
 *     2  Cholesky factorisation of a computed matrix: not positive definite
 *    -5  v5: workspace too small
 * (info != 0: the outputs are unspecified)
 */
extern "C" int64_t
estimate(double *v1, double *v2, size_t v3, int8_t *v4, int64_t *v5)
{
    int64_t v6 = 38737600;
    double v7 = 1.00000000000000000e+00;
    double v8 = 0.0e+00;
    int64_t v9 = 0;
    size_t v10 = 0;
    int64_t v11 = v5[v10];
    int64_t v12;
    if (v11 == -1) {
	v5[v10] = v6;
	v12 = v9;
    } else {
	int64_t v13;
	if (v11 > v9 & v11 < v6) {
	    v13 = -5;
	} else {
	    bool v14 = v11 == v9;
	    int8_t *v15;
	    if (v14) {
		void *v16 = aligned_alloc(64, 38737600);
		v15 = (int8_t *)v16;
	    } else {
		v15 = v4;
	    }
	    int8_t *v17 = v15;
	    int64_t v18 = (int64_t)v10;
	    double *v19 = (double *)(v17 + v18);
	    ulm_lam_copy(v19, v2, 32000000);
	    int64_t v20 =
	        ulmBLAS::potrf<std::ptrdiff_t>(true, 2000, v19, 2000, 1);
	    bool v21 = (bool)(v20 != v9);
	    uint64_t v22 = (uint64_t)v21;
	    int64_t v23 = (int64_t)v22;
	    int64_t v24;
	    if (v23 == v9) {
		int64_t v25 = (int64_t)32000000;
		double *v26 = (double *)(v17 + v25);
		int64_t v27 = (int64_t)32320000;
		double *v28 = (double *)(v17 + v27);
		ulm_lam_copy(v28, v1, 3200000);
		(void)ulmBLAS::trtrs<std::ptrdiff_t>(true, false, false, false,
		                                     2000, 200, v19, 2000, 1,
		                                     v28, 200, 1);
		ulmBLAS::sylrk<std::ptrdiff_t>(200, 2000, v7, v28, 1, 200, v8,
		                               v26, 200, 1);
		int64_t v29 =
		    ulmBLAS::potrf<std::ptrdiff_t>(true, 200, v26, 200, 1);
		int64_t v30 = v29 != v9 ? 2 : v9;
		if (v30 == v9) {
		    ptrdiff_t v31 = (ptrdiff_t)v3;
		    ptrdiff_t v32 = (ptrdiff_t)v10;
		    bool v33 = (v31 > v32 ? v3 : v10) * 8000000 + 960400000 <
		               (v31 > v32 ? v3 : v10) * 48800000;
		    int64_t v34 = (int64_t)35520000;
		    double *v35 = (double *)(v17 + v34);
		    if (v33) {
			ulm_lam_gecopy(200, 2000, v28, 1, 200, v35, 2000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    true, false, false, false, 200, 2000, v26, 200, 1,
			    v35, 2000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, 200, 2000, v26, 1, 200,
			    v35, 2000, 1);
			(void)ulmBLAS::trtrs<std::ptrdiff_t>(
			    false, false, false, false, 2000, 200, v19, 1, 2000,
			    v35, 1, 2000);
		    }
		    int64_t v36 = (int64_t)38720000;
		    double *v37 = (double *)(v17 + v36);
		    int64_t v38 = (int64_t)38736000;
		    double *v39 = (double *)(v17 + v38);
		    for (size_t i40 = v10; i40 < v3; i40 += 1) {
			observe(i40, v37);
			if (v33) {
			    ulmBLAS::gemv<std::ptrdiff_t>(200, 2000, v7, v35,
			                                  2000, 1, v37, 1, v8,
			                                  v39, 1);
			} else {
			    ulmBLAS::trlsv<std::ptrdiff_t>(2000, false, v19,
			                                   2000, 1, v37, 1);
			    ulmBLAS::gemv<std::ptrdiff_t>(
			        200, 2000, v7, v28, 1, 200, v37, 1, v8, v39, 1);
			    ulmBLAS::trlsv<std::ptrdiff_t>(200, false, v26, 200,
			                                   1, v39, 1);
			    ulmBLAS::trusv<std::ptrdiff_t>(200, false, v26, 1,
			                                   200, v39, 1);
			}
			record(i40, v39);
		    }
		}
		v24 = v30;
	    } else {
		v24 = v23;
	    }
	    int64_t v41 = v24;
	    if (v14) {
		free(v17);
	    }
	    v13 = v41;
	}
	v12 = v13;
    }
    return v12;
}
