//
//  driver.cc - workspaces of functions that call functions with
//  workspaces (compose.mlir).
//
//  postprocess(i, b, work, workspace_info) is an external function with a
//  workspace (it needs M doubles of scratch memory): it follows the
//  convention itself.  Checks:
//    * the query of estimate_all returns its own requirement plus the
//      answers of gls and postprocess (each rounded up to 64 bytes);
//    * with a given workspace, with *workspace_info = 0 (estimate_all
//      allocates) and per step, the estimates equal those of separate calls
//      of the generated gls;
//    * every call of postprocess got a part of estimate_all's workspace that
//      is large enough, and estimate_all leaves *workspace_info unchanged;
//    * a workspace that is too small calls ulm_lam_workspace_error and
//      computes nothing.
//
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <ulmblas/ulmblas.h>

#include "../workspace.h"

constexpr int N = 2000, M = 200, STEPS = 3;

extern "C" {
void
gls(double *X, double *M, double *y, double *b, int8_t *work,
    int64_t *workspace_info); // generated
void
estimate_all(double *X, double *M, size_t n, int8_t *work,
             int64_t *workspace_info); // generated
void
observe(size_t i, double *y);
void
postprocess(size_t i, double *b, int8_t *work, int64_t *workspace_info);
void
ulm_lam_workspace_error(int64_t required, int64_t given);
}

static std::vector<double>
X(N *M), Mat(N *N), noise(N *STEPS);
static std::vector<double>
results(M *STEPS);
static int errors = 0, badSlots = 0, calls = 0;
static int8_t *callerWork = nullptr;
static int64_t callerSize = 0;

void
observe(size_t i, double *y)
{
    std::vector<double> x(M);
    for (int j = 0; j < M; ++j)
	x[j] = std::sin(0.1 * i + 0.05 * j);
    ulmBLAS::gemv(N, M, 1.0, X.data(), M, 1, x.data(), 1, 0.0, y, 1);
    for (int r = 0; r < N; ++r)
	y[r] += noise[i * N + r];
}

//  An external function with a workspace: keeps b_i in results.
void
postprocess(size_t i, double *b, int8_t *work, int64_t *workspace_info)
{
    const int64_t required = M * sizeof(double);
    if (*workspace_info == -1) {
	*workspace_info = required;
	return;
    }
    ++calls;
    if (*workspace_info < required) {
	++badSlots;
	return;
    }
    //  the part must lie inside the caller's workspace (when given)
    if (callerWork &&
        (work < callerWork || work + required > callerWork + callerSize))
	++badSlots;
    double *tmp = reinterpret_cast<double *>(work);
    std::memcpy(tmp, b, required);
    std::memcpy(results.data() + i * M, tmp, required);
}

//  replaces the weak default (which aborts): count and return
void
ulm_lam_workspace_error(int64_t, int64_t)
{
    ++errors;
}

static int failures = 0;

static void
check(bool ok, const char *what)
{
    failures += !ok;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
}

int
main()
{
    std::mt19937_64 gen(9);
    std::uniform_real_distribution<double> rnd(-1, 1);
    std::vector<double> A(N * N);
    for (double &x : X)
	x = rnd(gen);
    for (double &x : A)
	x = rnd(gen);
    ulmBLAS::gemm(N, N, N, 1.0 / N, false, A.data(), N, 1, false, A.data(), 1,
                  N, 0.0, Mat.data(), N, 1);
    for (int i = 0; i < N; ++i)
	Mat[i * N + i] += 1.0;
    for (double &x : noise)
	x = 0.1 * rnd(gen);

    //  the accumulated query
    int64_t qAll = -1, qGls = -1, qPost = -1;
    estimate_all(nullptr, nullptr, STEPS, nullptr, &qAll);
    gls(nullptr, nullptr, nullptr, nullptr, nullptr, &qGls);
    postprocess(0, nullptr, nullptr, &qPost);
    auto up = [](int64_t x) { return (x + 63) / 64 * 64; };
    int64_t own = up(N * sizeof(double)) + up(M * sizeof(double)); // y, b
    std::printf("query: estimate_all %lld = own %lld + gls %lld + "
                "postprocess %lld\n",
                (long long)qAll, (long long)own, (long long)qGls,
                (long long)qPost);
    check(qAll == own + up(qGls) + up(qPost),
          "the query of estimate_all accumulates those of gls and "
          "postprocess");

    //  reference: separate calls of the generated gls
    std::vector<double> ref(M * STEPS), y(N);
    {
	Workspace ws([&](int8_t *w, int64_t *i) {
	    gls(X.data(), Mat.data(), y.data(), ref.data(), w, i);
	});
	for (int i = 0; i < STEPS; ++i) {
	    observe(i, y.data());
	    gls(X.data(), Mat.data(), y.data(), ref.data() + i * M, ws.work,
	        ws.info());
	}
    }
    auto same = [&] {
	double num = 0, den = 0;
	for (int e = 0; e < M * STEPS; ++e) {
	    num += (results[e] - ref[e]) * (results[e] - ref[e]);
	    den += ref[e] * ref[e];
	}
	return std::sqrt(num / den) < 1e-14;
    };

    //  a given workspace
    {
	Workspace ws([&](int8_t *w, int64_t *i) {
	    estimate_all(X.data(), Mat.data(), STEPS, w, i);
	});
	callerWork = ws.work;
	callerSize = ws.size;
	int64_t *info = ws.info();
	estimate_all(X.data(), Mat.data(), STEPS, ws.work, info);
	check(same(), "given workspace: estimates equal separate calls of gls");
	check(*info == ws.size,
	      "estimate_all leaves *workspace_info unchanged");
	check(calls == STEPS && badSlots == 0,
	      "every postprocess got a large enough part of the workspace");

	//  too small: error function, nothing computed
	std::fill(results.begin(), results.end(), 0.0);
	calls = errors = 0;
	int64_t small = ws.size - 1;
	estimate_all(X.data(), Mat.data(), STEPS, ws.work, &small);
	check(errors == 1 && calls == 0,
	      "workspace too small: error function called, nothing computed");
	callerWork = nullptr;
    }

    //  *workspace_info = 0: estimate_all allocates one block
    std::fill(results.begin(), results.end(), 0.0);
    calls = badSlots = 0;
    int64_t zero = 0;
    estimate_all(X.data(), Mat.data(), STEPS, nullptr, &zero);
    check(same() && badSlots == 0 && zero == 0,
          "workspace_info = 0: estimates equal separate calls of gls");

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
