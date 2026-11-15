//
//  driver.cc - check the functions generated from cases.mlir against naive
//  evaluations (triple loops, Gaussian elimination with partial pivoting,
//  substitution).  Row-major, n = 64, k = 32.
//
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../workspace.h"

constexpr int n = 64, k = 32;
using Mat = std::vector<double>;

//  every function returns info (0: success, k > 0: check k failed, -i:
//  argument i illegal)
//  f may overwrite its matrix arguments (LAPACK style); f_keep, the
//  wrapper ulm.lam adds, leaves them as they are (copies in its
//  workspace).  The checks call the wrappers, and lu itself once.
extern "C" {
int64_t
lu(double *A, double *B, double *X, int8_t *work, int64_t *workspace_info);
int64_t
lu_t(double *A, double *B, double *X, int8_t *work, int64_t *workspace_info);
int64_t
upper(double *U, double *B, double *X);
int64_t
right(double *B, double *L, double *X);
int64_t
right_t(double *B, double *L, double *X);
int64_t
chol_vec(double *S, double *x, double *r, int8_t *work,
         int64_t *workspace_info);
int64_t
sum(double *Y, double *Z, double *W, double *r);
int64_t
mixed(double *A, double *B, double *C, double *D, double *r, int8_t *work,
      int64_t *workspace_info);
int64_t
lu_keep(double *A, double *B, double *X, int8_t *work, int64_t *workspace_info);
int64_t
lu_t_keep(double *A, double *B, double *X, int8_t *work,
          int64_t *workspace_info);
int64_t
upper_keep(double *U, double *B, double *X);
int64_t
right_keep(double *B, double *L, double *X);
int64_t
right_t_keep(double *B, double *L, double *X);
int64_t
chol_vec_keep(double *S, double *x, double *r, int8_t *work,
              int64_t *workspace_info);
int64_t
sum_keep(double *Y, double *Z, double *W, double *r);
int64_t
mixed_keep(double *A, double *B, double *C, double *D, double *r, int8_t *work,
           int64_t *workspace_info);
}

static std::mt19937_64 gen(1);

static Mat
random(int r, int c)
{
    std::uniform_real_distribution<double> d(-1, 1);
    Mat m(r * c);
    for (double &x : m)
	x = d(gen);
    return m;
}

static Mat
mul(const Mat &A, const Mat &B, int m, int l, int c)
{
    Mat C(m * c, 0.0);
    for (int i = 0; i < m; ++i)
	for (int p = 0; p < l; ++p)
	    for (int j = 0; j < c; ++j)
		C[i * c + j] += A[i * l + p] * B[p * c + j];
    return C;
}

static Mat
transpose(const Mat &A, int r, int c)
{
    Mat T(r * c);
    for (int i = 0; i < r; ++i)
	for (int j = 0; j < c; ++j)
	    T[j * r + i] = A[i * c + j];
    return T;
}

//  A^-1 B by Gaussian elimination with partial pivoting (A is n x n)
static Mat
solve(Mat A, Mat B, int c)
{
    for (int j = 0; j < n; ++j) {
	int p = j;
	for (int i = j + 1; i < n; ++i)
	    if (std::fabs(A[i * n + j]) > std::fabs(A[p * n + j]))
		p = i;
	for (int q = 0; q < n; ++q)
	    std::swap(A[j * n + q], A[p * n + q]);
	for (int q = 0; q < c; ++q)
	    std::swap(B[j * c + q], B[p * c + q]);
	for (int i = j + 1; i < n; ++i) {
	    double f = A[i * n + j] / A[j * n + j];
	    for (int q = j; q < n; ++q)
		A[i * n + q] -= f * A[j * n + q];
	    for (int q = 0; q < c; ++q)
		B[i * c + q] -= f * B[j * c + q];
	}
    }
    for (int j = n - 1; j >= 0; --j)
	for (int q = 0; q < c; ++q) {
	    double s = B[j * c + q];
	    for (int i = j + 1; i < n; ++i)
		s -= A[j * n + i] * B[i * c + q];
	    B[j * c + q] = s / A[j * n + j];
	}
    return B;
}

static int failures = 0;
static int64_t info = 0; // of the last call

static void
expect(bool ok, const char *what)
{
    failures += !ok;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
}

static void
check(const char *name, const Mat &x, const Mat &ref)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < x.size(); ++i) {
	num += (x[i] - ref[i]) * (x[i] - ref[i]);
	den += ref[i] * ref[i];
    }
    double rel = std::sqrt(num / den);
    bool ok = rel < 1e-10 && info == 0;
    failures += !ok;
    std::printf("%s  %-9s relative error %.2e, info %lld\n",
                ok ? "ok  " : "FAIL", name, rel, (long long)info);
}

int
main()
{
    Mat A = random(n, n), B = random(n, k), Bk = random(k, n), C = random(n, k),
        D = random(n, k), x = random(n, 1), Y = random(n, n), Z = random(n, n),
        W = random(n, n);
    for (int i = 0; i < n; ++i)
	A[i * n + i] += n; // well conditioned
    Mat L = random(n, n), U = random(n, n);
    for (int i = 0; i < n; ++i)
	for (int j = 0; j < n; ++j) {
	    if (j > i)
		L[i * n + j] = 0;
	    if (j < i)
		U[i * n + j] = 0;
	}
    for (int i = 0; i < n; ++i) {
	L[i * n + i] += n;
	U[i * n + i] += n;
    }
    Mat S = mul(A, transpose(A, n, n), n, n, n); // spd

    Mat r(n * k), rk(k * n), rv(n), rs(n * n);
    int64_t zero = 0;
    info = lu_keep(A.data(), B.data(), r.data(), nullptr, &zero);
    check("lu", r, solve(A, B, k));
    info = lu_t_keep(A.data(), B.data(), r.data(), nullptr, &(zero = 0));
    check("lu_t", r, solve(transpose(A, n, n), B, k));
    info = upper_keep(U.data(), B.data(), r.data());
    check("upper", r, solve(U, B, k));
    //  B L^-1 = (L^-T B^T)^T
    info = right_keep(Bk.data(), L.data(), rk.data());
    check("right", rk,
          transpose(solve(transpose(L, n, n), transpose(Bk, k, n), k), n, k));
    info = right_t_keep(Bk.data(), L.data(), rk.data());
    check("right_t", rk, transpose(solve(L, transpose(Bk, k, n), k), n, k));
    info = chol_vec_keep(S.data(), x.data(), rv.data(), nullptr, &(zero = 0));
    check("chol_vec", rv, solve(S, x, 1));
    info = sum_keep(Y.data(), Z.data(), W.data(), rs.data());
    Mat ref(n * n);
    for (int i = 0; i < n * n; ++i)
	ref[i] = Y[i] + 2 * Z[i] - W[i];
    check("sum", rs, ref);
    Mat Bn = random(n, n);
    info = mixed_keep(A.data(), Bn.data(), C.data(), D.data(), r.data(),
                      nullptr, &(zero = 0));
    Mat m = solve(A, mul(Bn, C, n, n, k), k);
    for (int i = 0; i < n * k; ++i)
	m[i] += D[i];
    check("mixed", r, m);

    //  inputs passed by value must be unchanged
    bool unchanged = true;
    Mat A0 = A, S0 = S;
    lu_keep(A.data(), B.data(), r.data(), nullptr, &(zero = 0));
    chol_vec_keep(S.data(), x.data(), rv.data(), nullptr, &(zero = 0));
    unchanged = A == A0 && S == S0;
    failures += !unchanged;
    std::printf("%s  inputs passed by value are not overwritten\n",
                unchanged ? "ok  " : "FAIL");
    //  the workspace convention, with lu
    {
	Mat ref = solve(A, B, k), x(n * k, 0.0);
	int64_t q = -1;
	lu_keep(A.data(), B.data(), x.data(), nullptr, &q);
	bool queried = q > 0 && x == Mat(n * k, 0.0);
	failures += !queried;
	std::printf("%s  workspace query: %lld bytes, nothing computed\n",
	            queried ? "ok  " : "FAIL", (long long)q);
	Workspace ws([&](int8_t *w, int64_t *i) {
	    lu_keep(A.data(), B.data(), x.data(), w, i);
	});
	info = lu_keep(A.data(), B.data(), x.data(), ws.work, ws.info());
	check("lu, given workspace", x, ref);
	std::fill(x.begin(), x.end(), 0.0);
	int64_t small = ws.size - 1;
	info = lu_keep(A.data(), B.data(), x.data(), ws.work, &small);
	expect(info == -5 && x == Mat(n * k, 0.0),
	       "workspace too small: info = -5 (argument 5), nothing computed");
    }
    //  lu itself: A overwritten by its LU factors, no copy of A
    {
	Mat ref = solve(A, B, k), x(n * k, 0.0), Ac(A);
	int64_t q = -1, qKeep = -1;
	lu(Ac.data(), B.data(), x.data(), nullptr, &q);
	lu_keep(A.data(), B.data(), x.data(), nullptr, &qKeep);
	info = lu(Ac.data(), B.data(), x.data(), nullptr, &(zero = 0));
	check("lu itself", x, ref);
	expect(
	    Ac != A && q < qKeep,
	    "lu itself overwrites A, its workspace is smaller than lu_keep's");
    }
    //  the assumptions did not hold: info, no abort
    {
	Mat N(S);
	for (int i = 0; i < n; ++i)
	    N[i * n + i] = -1; // symmetric, indefinite
	info =
	    chol_vec_keep(N.data(), x.data(), rv.data(), nullptr, &(zero = 0));
	std::printf("      chol_vec of an indefinite matrix: info %lld\n",
	            (long long)info);
	expect(info == 1, "not positive definite: info = 1 (factorisation 1)");
	Mat Z0(n * n, 0.0); // singular
	info = lu_keep(Z0.data(), B.data(), r.data(), nullptr, &(zero = 0));
	expect(info == 1, "singular: info = 1 (LU factorisation)");
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
