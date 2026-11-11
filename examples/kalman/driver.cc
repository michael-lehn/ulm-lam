//
//  driver.cc - run estimate() generated from estimate.mlir and check it.
//
//  observe(i, y) produces the measurement of step i: y_i = H x_i + e_i with
//  a slowly moving state x_i and correlated noise e_i.  record(i, b) keeps
//  the estimate b_i.  Three versions are compared:
//
//    estimate         the ulm.lam pipeline
//    estimate_nolicm  the same without loop-invariant-code-motion: R is
//                     still factorised once (it is a function argument),
//                     but S = H^T R^-1 H is formed and factorised in every
//                     step
//    explicit         formula (1) from scratch in every step, with
//                     explicit inverses (getrf + getri): the literal
//                     formula, the same input ulm.lam gets
//    explicit, hoisted  hand-written: the invariant inverses once before
//                     the loop, two gemv per step
//    operator G       hand-written: G = (H^T R^-1 H)^-1 H^T R^-1 once,
//                     one gemv per step (fewest FLOPs per step; pays off
//                     for many steps)
//
//  Every estimate is checked through the normal equations
//  H^T R^-1 (H b_i - y_i) = 0 with an independent LU solve, and the three
//  versions must agree.
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/getrs.h>
#include <ulmblas/ulmblas.h>

#include "../workspace.h"

constexpr int N = 2000, M = 200, STEPS = 20;
//  timing: the steps of a timed run, best of REPS (the literal formula is
//  timed with STEPS steps only: it takes about 0.1 s per step)
constexpr int TSTEPS = 200, REPS = 3;

extern "C" {
void
estimate(double *H, double *R, size_t n, int8_t *work,
         int64_t *workspace_info); // generated
void
estimate_nolicm(double *H, double *R, size_t n, int8_t *work,
                int64_t *workspace_info); // generated, no LICM
void
observe(size_t i, double *y);
void
record(size_t i, double *b);
}

static std::vector<double>
H(N *M), R(N *N), noise(N *TSTEPS);
static std::vector<double>
recorded(M *TSTEPS);

//  y_i = H x_i + e_i, x_i(j) = sin(0.1 i + 0.05 j)
void
observe(size_t i, double *y)
{
    std::vector<double> x(M);
    for (int j = 0; j < M; ++j)
	x[j] = std::sin(0.1 * i + 0.05 * j);
    ulmBLAS::gemv(N, M, 1.0, H.data(), M, 1, x.data(), 1, 0.0, y, 1);
    for (int r = 0; r < N; ++r)
	y[r] += noise[i * N + r];
}

void
record(size_t i, double *b)
{
    std::copy(b, b + M, recorded.begin() + i * M);
}

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

//  formula (1) from scratch in every step, with explicit inverses
static void
estimateExplicit(size_t n)
{
    std::vector<double> Ri(N * N), T(M * N), S(M * M), t(M), y(N), b(M);
    std::vector<int> piv(N);
    for (size_t i = 0; i < n; ++i) {
	observe(i, y.data());
	Ri = R;
	ulmBLAS::getrf(N, N, Ri.data(), N, 1, piv.data(), 1);
	ulmBLAS::getri(N, Ri.data(), N, 1, piv.data(), 1);
	ulmBLAS::gemm(M, N, N, 1.0, false, H.data(), 1, M, false, Ri.data(), N,
	              1, 0.0, T.data(), N, 1); // H^T R^-1
	ulmBLAS::gemm(M, M, N, 1.0, false, T.data(), N, 1, false, H.data(), M,
	              1, 0.0, S.data(), M, 1);
	ulmBLAS::getrf(M, M, S.data(), M, 1, piv.data(), 1);
	ulmBLAS::getri(M, S.data(), M, 1, piv.data(), 1);
	ulmBLAS::gemv(M, N, 1.0, T.data(), N, 1, y.data(), 1, 0.0, t.data(), 1);
	ulmBLAS::gemv(M, M, 1.0, S.data(), M, 1, t.data(), 1, 0.0, b.data(), 1);
	record(i, b.data());
    }
}

//  hand-written, hoisted: the invariant explicit inverses once, before the
//  loop (getrf + getri), then H^T R^-1 y_i and S^-1 (.) per step
static void
estimateExplicitHoisted(size_t n)
{
    std::vector<double> Ri(R), T(M * N), S(M * M), t(M), y(N), b(M);
    std::vector<int> piv(N);
    ulmBLAS::getrf(N, N, Ri.data(), N, 1, piv.data(), 1);
    ulmBLAS::getri(N, Ri.data(), N, 1, piv.data(), 1);
    ulmBLAS::gemm(M, N, N, 1.0, false, H.data(), 1, M, false, Ri.data(), N, 1,
                  0.0, T.data(), N, 1); // T = H^T R^-1
    ulmBLAS::gemm(M, M, N, 1.0, false, T.data(), N, 1, false, H.data(), M, 1,
                  0.0, S.data(), M, 1);
    ulmBLAS::getrf(M, M, S.data(), M, 1, piv.data(), 1);
    ulmBLAS::getri(M, S.data(), M, 1, piv.data(), 1); // S^-1
    for (size_t i = 0; i < n; ++i) {
	observe(i, y.data());
	ulmBLAS::gemv(M, N, 1.0, T.data(), N, 1, y.data(), 1, 0.0, t.data(), 1);
	ulmBLAS::gemv(M, M, 1.0, S.data(), M, 1, t.data(), 1, 0.0, b.data(), 1);
	record(i, b.data());
    }
}

//  hand-written, best for many steps: the whole operator
//  G = (H^T R^-1 H)^-1 H^T R^-1 (m x n) once, then one gemv per step
static void
estimateOperator(size_t n)
{
    std::vector<double> Ri(R), T(M * N), S(M * M), G(M * N), y(N), b(M);
    std::vector<int> piv(N);
    ulmBLAS::getrf(N, N, Ri.data(), N, 1, piv.data(), 1);
    ulmBLAS::getri(N, Ri.data(), N, 1, piv.data(), 1);
    ulmBLAS::gemm(M, N, N, 1.0, false, H.data(), 1, M, false, Ri.data(), N, 1,
                  0.0, T.data(), N, 1);
    ulmBLAS::gemm(M, M, N, 1.0, false, T.data(), N, 1, false, H.data(), M, 1,
                  0.0, S.data(), M, 1);
    ulmBLAS::getrf(M, M, S.data(), M, 1, piv.data(), 1);
    ulmBLAS::getri(M, S.data(), M, 1, piv.data(), 1);
    ulmBLAS::gemm(M, N, M, 1.0, false, S.data(), M, 1, false, T.data(), N, 1,
                  0.0, G.data(), N, 1); // G = S^-1 T
    for (size_t i = 0; i < n; ++i) {
	observe(i, y.data());
	ulmBLAS::gemv(M, N, 1.0, G.data(), N, 1, y.data(), 1, 0.0, b.data(), 1);
	record(i, b.data());
    }
}

static double
nrm2(const double *v, int n, int inc = 1)
{
    double s = 0;
    for (int i = 0; i < n; ++i)
	s += v[i * inc] * v[i * inc];
    return std::sqrt(s);
}

int
main()
{
    std::mt19937_64 gen(3);
    std::uniform_real_distribution<double> rnd(-1, 1);
    std::vector<double> A(N * N);
    for (double &x : H)
	x = rnd(gen);
    for (double &x : A)
	x = rnd(gen);
    ulmBLAS::gemm(N, N, N, 1.0 / N, false, A.data(), N, 1, false, A.data(), 1,
                  N, 0.0, R.data(), N, 1); // A A^T / N: correlated
    for (int i = 0; i < N; ++i)
	R[i * N + i] += 1.0;
    for (double &x : noise)
	x = 0.1 * rnd(gen);

    //  every version: STEPS steps for the checks; timed with `steps` steps
    //  and with 0 steps (its one-time part), best of REPS
    struct Run
    {
	const char *name;
	int steps;
	double t, setup;
	std::vector<double> b;
	double
	perStep() const
	{
	    return (t - setup) / steps;
	}
	double
	at(int n) const
	{
	    return setup + n * perStep();
	}
    };
    std::vector<Run> runs;
    auto best = [&](auto f, size_t n) {
	double t = 1e30;
	for (int r = 0; r < REPS; ++r) {
	    double t0 = now();
	    f(n);
	    t = std::min(t, now() - t0);
	}
	return t;
    };
    auto run = [&](const char *name, auto f, int steps = TSTEPS) {
	f(STEPS);
	std::vector<double> b(recorded.begin(), recorded.begin() + M * STEPS);
	double t = best(f, steps), setup = best(f, 0);
	runs.push_back({name, steps, t, setup, b});
    };
    Workspace ws(
        [](int8_t *w, int64_t *i) { estimate(H.data(), R.data(), 0, w, i); });
    Workspace wsNoLicm([](int8_t *w, int64_t *i) {
	estimate_nolicm(H.data(), R.data(), 0, w, i);
    });
    std::printf("workspace: %lld bytes (without LICM: %lld)\n",
                (long long)ws.size, (long long)wsNoLicm.size);
    run("ulm.lam",
        [&](size_t n) { estimate(H.data(), R.data(), n, ws.work, ws.info()); });
    run("ulm.lam without LICM", [&](size_t n) {
	estimate_nolicm(H.data(), R.data(), n, wsNoLicm.work, wsNoLicm.info());
    });
    run("explicit inverses", [](size_t n) { estimateExplicit(n); }, STEPS);
    run("explicit, hoisted", [](size_t n) { estimateExplicitHoisted(n); });
    run("operator G, hoisted", [](size_t n) { estimateOperator(n); });

    //  normal equations of every step for the ulm.lam result, R^-1 by LU
    std::vector<double> LU(R), U(N * 2 * STEPS);
    std::vector<int> piv(N);
    const std::vector<double> &b = runs[0].b;
    std::vector<double> y(N);
    for (int i = 0; i < STEPS; ++i) {
	observe(i, y.data());
	for (int r = 0; r < N; ++r) {
	    double hb = 0;
	    for (int j = 0; j < M; ++j)
		hb += H[r * M + j] * b[i * M + j];
	    U[r * 2 * STEPS + 2 * i] = hb - y[r];
	    U[r * 2 * STEPS + 2 * i + 1] = y[r];
	}
    }
    ulmBLAS::getrf(N, N, LU.data(), N, 1, piv.data(), 1);
    ulmBLAS::getrs(false, false, N, 2 * STEPS, LU.data(), N, 1, piv.data(), 1,
                   U.data(), 2 * STEPS, 1);
    double worst = 0;
    std::vector<double> rr(M), q(M);
    for (int i = 0; i < STEPS; ++i) {
	ulmBLAS::gemv(M, N, 1.0, H.data(), 1, M, U.data() + 2 * i, 2 * STEPS,
	              0.0, rr.data(), 1);
	ulmBLAS::gemv(M, N, 1.0, H.data(), 1, M, U.data() + 2 * i + 1,
	              2 * STEPS, 0.0, q.data(), 1);
	worst = std::max(worst, nrm2(rr.data(), M) / nrm2(q.data(), M));
    }
    double diff = 0;
    for (size_t k = 1; k < runs.size(); ++k) {
	double num = 0, den = 0;
	for (int e = 0; e < M * STEPS; ++e) {
	    num += (runs[k].b[e] - b[e]) * (runs[k].b[e] - b[e]);
	    den += b[e] * b[e];
	}
	diff = std::max(diff, std::sqrt(num / den));
    }

    std::printf("n = %d sensors, m = %d states; %d steps = one-time part + "
                "%d x time per step (both measured)\n",
                N, M, STEPS, STEPS);
    std::printf("%-20s  %10s  %12s  %10s  %7s\n", "", "total [ms]",
                "before [ms]", "step [ms]", "vs ulm.lam");
    for (const Run &r : runs)
	std::printf("%-20s  %10.2f  %12.2f  %10.3f  %6.1fx\n", r.name,
	            1e3 * r.at(STEPS), 1e3 * r.setup, 1e3 * r.perStep(),
	            r.at(STEPS) / runs[0].at(STEPS));
    //  a version that pays more before the loop and less per step is
    //  faster than ulm.lam from this number of steps on
    for (const Run &r : runs)
	if (r.perStep() < runs[0].perStep() && r.setup > runs[0].setup)
	    std::printf(
	        "%s is faster than ulm.lam from about %.0f steps on\n", r.name,
	        (r.setup - runs[0].setup) / (runs[0].perStep() - r.perStep()));
    std::printf("worst normal-equation residual over all steps = %.2e\n",
                worst);
    std::printf("largest relative difference between the versions = %.2e\n",
                diff);
    bool ok = worst < 1e-10 && diff < 1e-8;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
