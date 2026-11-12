//
//  driver_chain.cc - D = A B C with sizes known only at run time
//  (chain.mlir).  The generated chain() computes the costs of (A B) C and
//  A (B C) from n, k, m, p and runs the cheaper one.  For sizes where
//  either plan wins:
//
//  * D against a naive triple loop;
//  * the workspace query: the temporary of the plan that runs, A B
//    (n x m) or B C (k x p), rounded up to 64 bytes -- not both.
//
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../workspace.h"

extern "C" int64_t
chain(size_t n, size_t k, size_t m, size_t p, double *A, double *B, double *C,
      double *D, int8_t *work, int64_t *workspace_info); // generated

static int64_t
round64(int64_t bytes)
{
    return (bytes + 63) / 64 * 64;
}

static bool
check(int n, int k, int m, int p, std::mt19937_64 &gen)
{
    std::uniform_real_distribution<double> rnd(-1, 1);
    std::vector<double> A(n * k), B(k * m), C(m * p), D(n * p), R(n * p, 0.0),
        T(n * m, 0.0);
    for (double &x : A)
	x = rnd(gen);
    for (double &x : B)
	x = rnd(gen);
    for (double &x : C)
	x = rnd(gen);
    Workspace ws([&](int8_t *w, int64_t *i) {
	chain(n, k, m, p, A.data(), B.data(), C.data(), D.data(), w, i);
    });
    int64_t info = chain(n, k, m, p, A.data(), B.data(), C.data(), D.data(),
                         ws.work, ws.info());
    //  reference: (A B) C by loops
    for (int i = 0; i < n; ++i)
	for (int l = 0; l < k; ++l)
	    for (int j = 0; j < m; ++j)
		T[i * m + j] += A[i * k + l] * B[l * m + j];
    for (int i = 0; i < n; ++i)
	for (int l = 0; l < m; ++l)
	    for (int j = 0; j < p; ++j)
		R[i * p + j] += T[i * m + l] * C[l * p + j];
    double num = 0, den = 0;
    for (int i = 0; i < n * p; ++i) {
	num += (D[i] - R[i]) * (D[i] - R[i]);
	den += R[i] * R[i];
    }
    double rel = std::sqrt(num / den);
    bool left = double(n) * k * m + double(n) * m * p <=
                double(k) * m * p + double(n) * k * p;
    int64_t expected = left ? round64(8 * n * m) : round64(8 * k * p);
    bool ok = info == 0 && rel < 1e-12 && ws.size == expected;
    std::printf("n %4d k %4d m %4d p %4d: %s, workspace %8lld (expected "
                "%8lld), error %.1e  %s\n",
                n, k, m, p, left ? "(A B) C" : "A (B C)", (long long)ws.size,
                (long long)expected, rel, ok ? "ok" : "FAIL");
    return ok;
}

int
main()
{
    std::mt19937_64 gen(5);
    bool ok = true;
    for (auto [n, k, m, p] :
         {std::array{300, 10, 300, 10}, std::array{10, 300, 10, 300},
          std::array{200, 200, 5, 200}, std::array{5, 200, 200, 200},
          std::array{1, 1, 1, 1}})
	ok &= check(n, k, m, p, gen);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
