/* driver.c - run the C code generated from accumulate.mlir and check it.
 *
 *   for i = 0, ..., n-1:   B = foo(i);   X += A * B * C
 *
 * accumulate() is the generated code (C++ calling ulmBLAS).  The reference
 * evaluates the same loop with naive triple loops.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    N = 500,
    ITERS = 4
};

int64_t
accumulate(double *A, double *C, double *X, size_t n, int8_t *work,
           int64_t *workspace_info); /* generated, returns info */

/* The external function, called as foo(i, B) with B by reference. */
void
foo(size_t i, double *B)
{
    for (size_t r = 0; r < N; ++r)
	for (size_t c = 0; c < N; ++c)
	    B[r * N + c] =
	        sin(0.1 * (double)i + 0.01 * (double)r - 0.003 * (double)c);
}

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void
matmul_acc(const double *P, const double *Q, double *R, int zero)
{
    if (zero)
	memset(R, 0, sizeof(double) * N * N);
    for (size_t r = 0; r < N; ++r)
	for (size_t k = 0; k < N; ++k) {
	    double p = P[r * N + k];
	    for (size_t c = 0; c < N; ++c)
		R[r * N + c] += p * Q[k * N + c];
	}
}

int
main(void)
{
    double *A = malloc(sizeof(double) * N * N),
           *C = malloc(sizeof(double) * N * N);
    double *X = malloc(sizeof(double) * N * N),
           *Xr = malloc(sizeof(double) * N * N);
    double *B = malloc(sizeof(double) * N * N),
           *T = malloc(sizeof(double) * N * N);
    srand(7);
    for (size_t i = 0; i < (size_t)N * N; ++i) {
	A[i] = 2.0 * rand() / RAND_MAX - 1.0;
	C[i] = 2.0 * rand() / RAND_MAX - 1.0;
	X[i] = Xr[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }

    double t0 = now();
    /* the workspace convention: query the size, allocate, pass it */
    int64_t info = -1;
    accumulate(A, C, X, ITERS, NULL, &info);
    int8_t *work = aligned_alloc(64, (size_t)(info + 63) / 64 * 64);
    printf("workspace: %lld bytes\n", (long long)info);
    int64_t status = accumulate(A, C, X, ITERS, work, &info);
    free(work);
    double t_gen = now() - t0;

    t0 = now();
    for (size_t i = 0; i < ITERS; ++i) {
	foo(i, B);
	matmul_acc(A, B, T, 1);  /* T = A B   */
	matmul_acc(T, C, Xr, 0); /* X += T C  */
    }
    double t_ref = now() - t0;

    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
	num += (X[i] - Xr[i]) * (X[i] - Xr[i]);
	den += Xr[i] * Xr[i];
    }
    double rel = sqrt(num / den);
    printf("n = %d, %d iterations\n", N, ITERS);
    printf("generated C++, ulmBLAS: %8.2f ms\n", 1e3 * t_gen);
    printf("naive C loops        : %8.2f ms\n", 1e3 * t_ref);
    printf("||X - X_ref|| / ||X_ref|| = %.2e\n", rel);
    printf("info = %lld\n", (long long)status);
    int ok = rel < 1e-12 && status == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(A);
    free(C);
    free(X);
    free(Xr);
    free(B);
    free(T);
    return ok ? 0 : 1;
}
