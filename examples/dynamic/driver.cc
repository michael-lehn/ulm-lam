//
//  driver.cc - one compiled gls() (from gls.mlir, sizes as parameters) for
//  matrices of several sizes.  For every (n, m):
//
//  1. random X (n x m), M = A A^T + n I (spd), y
//  2. query the workspace; it must be the sum of the temporaries, each
//     rounded up to 64 bytes: L (n x n), the Cholesky factor of S (m x m),
//     W = L^-1 X (n x m), L^-1 y (n)
//  3. b = gls(n, m, X, M, y)
//  4. check: relative residual of the normal equations
//     X^T M^-1 (X b - y), with M^-1 applied by an independent LU solve
//     (getrf + getrs), and that X, M, y are unchanged (intent in)
//
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrs.h>
#include <ulmblas/ulmblas.h>

#include "../workspace.h"

extern "C" void
gls(size_t n, size_t m, double *X, double *M, double *y, double *b,
    int8_t *work, int64_t *workspace_info); // generated

static int64_t
round64(int64_t bytes)
{
    return (bytes + 63) / 64 * 64;
}

static double
nrm2(const std::vector<double> &v)
{
    double s = 0;
    for (double x : v)
	s += x * x;
    return std::sqrt(s);
}

static bool
check(int n, int m, std::mt19937_64 &gen)
{
    std::uniform_real_distribution<double> rnd(-1, 1);
    std::vector<double> X(n * m), M(n * n), A(n * n), y(n), b(m);
    for (double &x : X)
	x = rnd(gen);
    for (double &x : A)
	x = rnd(gen);
    for (double &x : y)
	x = rnd(gen);
    ulmBLAS::gemm(n, n, n, 1.0, false, A.data(), n, 1, false, A.data(), 1, n,
                  0.0, M.data(), n, 1); // A A^T
    for (int i = 0; i < n; ++i)
	M[i * n + i] += n;
    std::vector<double> X0(X), M0(M), y0(y);

    Workspace ws([&](int8_t *w, int64_t *i) {
	gls(n, m, X.data(), M.data(), y.data(), b.data(), w, i);
    });
    int64_t expected = round64(8 * n * n) + round64(8 * m * m) +
                       round64(8 * n * m) + round64(8 * n);
    gls(n, m, X.data(), M.data(), y.data(), b.data(), ws.work, ws.info());

    //  residual r = X^T M^-1 (X b - y), M^-1 by LU (independent of potrf)
    std::vector<double> U(n * 2), LU(M), r(m), q(m);
    std::vector<int> piv(n);
    for (int i = 0; i < n; ++i) {
	double xb = 0;
	for (int j = 0; j < m; ++j)
	    xb += X[i * m + j] * b[j];
	U[i * 2 + 0] = xb - y[i];
	U[i * 2 + 1] = y[i];
    }
    ulmBLAS::getrf(n, n, LU.data(), n, 1, piv.data(), 1);
    ulmBLAS::getrs(false, false, n, 2, LU.data(), n, 1, piv.data(), 1, U.data(),
                   2, 1);
    ulmBLAS::gemv(m, n, 1.0, X.data(), 1, m, U.data(), 2, 0.0, r.data(), 1);
    ulmBLAS::gemv(m, n, 1.0, X.data(), 1, m, U.data() + 1, 2, 0.0, q.data(),
                  1);
    double res = nrm2(r) / nrm2(q);
    bool unchanged = X == X0 && M == M0 && y == y0;
    bool ok = res < 1e-10 && ws.size == expected && unchanged;
    std::printf("n = %4d, m = %3d: workspace %9lld bytes (expected %9lld), "
                "residual %.1e, inputs %s  %s\n",
                n, m, (long long)ws.size, (long long)expected, res,
                unchanged ? "unchanged" : "CHANGED", ok ? "ok" : "FAIL");
    return ok;
}

int
main()
{
    std::mt19937_64 gen(42);
    bool ok = true;
    for (auto [n, m] : {std::pair{1, 1}, {7, 3}, {100, 1}, {100, 100},
                        {501, 37}, {2000, 200}})
	ok &= check(n, m, gen);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
