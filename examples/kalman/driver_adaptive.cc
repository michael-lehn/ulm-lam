//
//  driver_adaptive.cc - run estimate_adaptive() generated from
//  adaptive.mlir and check it.
//
//  update_covariance(i, R) updates the noise covariance in place: it moves
//  R towards a base covariance and changes the sensor noise level with the
//  step, R <- 0.9 R + 0.1 R_base + 0.05 (1 + sin(0.3 i)) I, which keeps R
//  spd.  observe(i, y) and record(i, b) as in driver.cc.  Compared with
//  formula (1) evaluated with explicit inverses in every step (getrf +
//  getri) on the same covariances; every estimate is checked through the
//  normal equations H^T R_i^-1 (H b_i - y_i) = 0 with an LU solve.
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

constexpr int N = 2000, M = 200, STEPS = 10;

extern "C" {
void
estimate_adaptive(double *H, double *R0, size_t n, int8_t *work,
                  int64_t *workspace_info); // generated
void
update_covariance(size_t i, double *R);
void
observe(size_t i, double *y);
void
record(size_t i, double *b);
}

static std::vector<double>
H(N *M), Rbase(N *N), noise(N *STEPS);
static std::vector<double>
recorded(M *STEPS);

void
update_covariance(size_t i, double *R)
{
    double level = 0.05 * (1 + std::sin(0.3 * i));
    for (int r = 0; r < N; ++r)
	for (int c = 0; c < N; ++c)
	    R[r * N + c] = 0.9 * R[r * N + c] + 0.1 * Rbase[r * N + c] +
	                   (r == c ? level : 0.0);
}

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

//  formula (1) with explicit inverses in every step, same covariances
static void
estimateExplicit(const std::vector<double> &R0, size_t n)
{
    std::vector<double> R(R0), Ri(N * N), T(M * N), S(M * M), t(M), y(N), b(M);
    std::vector<int> piv(N);
    for (size_t i = 0; i < n; ++i) {
	update_covariance(i, R.data());
	observe(i, y.data());
	Ri = R;
	ulmBLAS::getrf(N, N, Ri.data(), N, 1, piv.data(), 1);
	ulmBLAS::getri(N, Ri.data(), N, 1, piv.data(), 1);
	ulmBLAS::gemm(M, N, N, 1.0, false, H.data(), 1, M, false, Ri.data(), N,
	              1, 0.0, T.data(), N, 1);
	ulmBLAS::gemm(M, M, N, 1.0, false, T.data(), N, 1, false, H.data(), M,
	              1, 0.0, S.data(), M, 1);
	ulmBLAS::getrf(M, M, S.data(), M, 1, piv.data(), 1);
	ulmBLAS::getri(M, S.data(), M, 1, piv.data(), 1);
	ulmBLAS::gemv(M, N, 1.0, T.data(), N, 1, y.data(), 1, 0.0, t.data(), 1);
	ulmBLAS::gemv(M, M, 1.0, S.data(), M, 1, t.data(), 1, 0.0, b.data(), 1);
	record(i, b.data());
    }
}

int
main()
{
    std::mt19937_64 gen(5);
    std::uniform_real_distribution<double> rnd(-1, 1);
    std::vector<double> A(N * N), R0(N * N);
    for (double &x : H)
	x = rnd(gen);
    for (double &x : A)
	x = rnd(gen);
    ulmBLAS::gemm(N, N, N, 1.0 / N, false, A.data(), N, 1, false, A.data(), 1,
                  N, 0.0, Rbase.data(), N, 1);
    for (int i = 0; i < N; ++i)
	Rbase[i * N + i] += 1.0;
    R0 = Rbase;
    for (double &x : noise)
	x = 0.1 * rnd(gen);

    std::vector<double> R0copy(R0);
    Workspace ws([&](int8_t *w, int64_t *i) {
	estimate_adaptive(H.data(), R0.data(), 0, w, i);
    });
    std::printf("workspace: %lld bytes\n", (long long)ws.size);
    estimate_adaptive(H.data(), R0.data(), STEPS, ws.work,
                      ws.info()); // warm-up
    double t0 = now();
    estimate_adaptive(H.data(), R0.data(), STEPS, ws.work, ws.info());
    double tUlmLam = now() - t0;
    std::vector<double> bUlmLam(recorded);
    bool unchanged = R0 == R0copy;

    estimateExplicit(R0, STEPS); // warm-up
    t0 = now();
    estimateExplicit(R0, STEPS);
    double tExplicit = now() - t0;

    //  normal equations of every step, R_i replayed, R_i^-1 by LU
    std::vector<double> R(R0), LU(N * N), U(N * 2), y(N), rr(M), q(M);
    std::vector<int> piv(N);
    double worst = 0;
    for (int i = 0; i < STEPS; ++i) {
	update_covariance(i, R.data());
	observe(i, y.data());
	for (int r = 0; r < N; ++r) {
	    double hb = 0;
	    for (int j = 0; j < M; ++j)
		hb += H[r * M + j] * bUlmLam[i * M + j];
	    U[2 * r] = hb - y[r];
	    U[2 * r + 1] = y[r];
	}
	LU = R;
	ulmBLAS::getrf(N, N, LU.data(), N, 1, piv.data(), 1);
	ulmBLAS::getrs(false, false, N, 2, LU.data(), N, 1, piv.data(), 1,
	               U.data(), 2, 1);
	ulmBLAS::gemv(M, N, 1.0, H.data(), 1, M, U.data(), 2, 0.0, rr.data(),
	              1);
	ulmBLAS::gemv(M, N, 1.0, H.data(), 1, M, U.data() + 1, 2, 0.0, q.data(),
	              1);
	double a = 0, c = 0;
	for (int j = 0; j < M; ++j) {
	    a += rr[j] * rr[j];
	    c += q[j] * q[j];
	}
	worst = std::max(worst, std::sqrt(a / c));
    }
    double num = 0, den = 0;
    for (int e = 0; e < M * STEPS; ++e) {
	num += (recorded[e] - bUlmLam[e]) * (recorded[e] - bUlmLam[e]);
	den += bUlmLam[e] * bUlmLam[e];
    }
    double diff = std::sqrt(num / den);

    std::printf("n = %d sensors, m = %d states, %d steps, R updated in "
                "every step\n",
                N, M, STEPS);
    std::printf("ulm.lam              : %9.2f ms  (%6.2f ms per step)\n",
                1e3 * tUlmLam, 1e3 * tUlmLam / STEPS);
    std::printf("explicit inverses : %9.2f ms  (%6.2f ms per step, %.1fx)\n",
                1e3 * tExplicit, 1e3 * tExplicit / STEPS, tExplicit / tUlmLam);
    std::printf("worst normal-equation residual over all steps = %.2e\n",
                worst);
    std::printf("relative difference ulm.lam / explicit = %.2e\n", diff);
    std::printf("R0 passed by value is unchanged: %s\n",
                unchanged ? "yes" : "no");
    bool ok = worst < 1e-10 && diff < 1e-8 && unchanged;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
