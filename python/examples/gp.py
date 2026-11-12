"""Gaussian process (GP) regression with ulm.lam, on two public data sets.

A GP with kernel k predicts at test points X* from training data (X, y)

    mean = K*^T K^-1 y,    cov = K** - K*^T K^-1 K*

with K = k(X, X) + s^2 I (n x n, spd), K* = k(X, X*) (n x m) and
K** = k(X*, X*) (m x m).  This is the formula of every textbook.  Written
literally in NumPy it forms K^-1; libraries therefore implement it by hand,
e.g. scikit-learn's GaussianProcessRegressor (fit and predict):

    L = cholesky(K);  alpha = L^-T L^-1 y;  V = L^-1 K*
    mean = K*^T alpha;  cov = K** - V^T V

ulm.lam derives this from the formula (potrf, trsm, trsv, gemv and syrk;
K^-1 is never formed), for all n and m: the sizes are run-time names.

Data (downloaded on first use into build/data/):
  * CO2: weekly CO2 concentration at Mauna Loa, NOAA (National Oceanic and
    Atmospheric Administration) Global Monitoring Laboratory and Scripps
    Institution of Oceanography (gml.noaa.gov/ccgg/trends/).  The classic
    GP example (Rasmussen and Williams, Gaussian Processes for Machine
    Learning, 2006, Sec. 5.4.3): forecast with the kernel and the
    hyperparameters of the book.
  * Bike: hourly rentals of the Capital Bikeshare system, UCI (University
    of California, Irvine) Machine Learning Repository (Fanaee-T and Gama,
    Event labeling combining ensemble detectors and background knowledge,
    Progress in Artificial Intelligence, 2013).  Tabular regression with up
    to 16000 training hours: timings.

Not done by ulm.lam (yet): the kernel matrices (elementwise functions,
NumPy here), the noise term s^2 I (added in NumPy: needs the identity and
scalars known at run time) and fitting the hyperparameters (needs
log det K = 2 sum log L_ii and traces; scikit-learn here, on a subset).

    python3 python/examples/gp.py            # n up to 8000
    python3 python/examples/gp.py --full     # n up to 16000

Optional: scikit-learn (comparison, hyperparameters of the bike data),
SciPy (comparison), Matplotlib (build/gp/co2.png).
"""

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Matrix, Vector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "build" / "data"
OUT = ROOT / "build" / "gp"

try:
    from scipy.linalg import cho_solve, cholesky, solve_triangular
except ImportError:
    cholesky = None
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import kernels as skk
except ImportError:
    GaussianProcessRegressor = None


@lam.function
def gp(K: Matrix("n", "n", "spd"), Ks: Matrix("n", "m"),
       Kss: Matrix("m", "m", "symmetric"), y: Vector("n")):
    mean = Ks.T @ K.inv @ y
    cov = Kss - Ks.T @ K.inv @ Ks
    return mean, cov


# ---------------------------------------------------------------------------
# The same without ulm.lam
# ---------------------------------------------------------------------------


def numpy_literal(K, Ks, Kss, y):
    """The formula as written."""
    Ki = np.linalg.inv(K)
    return Ks.T @ Ki @ y, Kss - Ks.T @ Ki @ Ks


def scipy_by_hand(K, Ks, Kss, y):
    """What scikit-learn does inside fit and predict."""
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y)
    V = solve_triangular(L, Ks, lower=True)
    return Ks.T @ alpha, Kss - V.T @ V


def numpy_by_hand(K, Ks, Kss, y):
    """The same with NumPy only: it has no triangular solve,
    np.linalg.solve factorises the triangular L again (LU)."""
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    V = np.linalg.solve(L, Ks)
    return Ks.T @ alpha, Kss - V.T @ V


reference = scipy_by_hand if cholesky else numpy_by_hand


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def difference(got, ref):
    mean, cov = got
    return max(np.abs(mean.ravel() - ref[0].ravel()).max()
               / np.abs(ref[0]).max(),
               np.abs(cov - ref[1]).max() / np.abs(ref[1]).max())


def fetch(url, name):
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        tmp = path.with_name(name + ".partial")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# CO2 at Mauna Loa
# ---------------------------------------------------------------------------

# Rasmussen and Williams, eq. (5.15)-(5.18), hyperparameters fitted there
# (ppm, years): long-term trend, seasonal component, medium-term
# irregularities, correlated noise.  White noise: weekly instead of
# monthly means, so larger than in the book.
CO2_NOISE = 0.3


def co2_kernel(a, b):
    r = a[:, None] - b[None, :]
    trend = 66.0**2 * np.exp(-(r**2) / (2 * 67.0**2))
    season = 2.4**2 * np.exp(-(r**2) / (2 * 90.0**2)
                             - 2 * np.sin(np.pi * r) ** 2 / 1.3**2)
    medium = 0.66**2 * (1 + r**2 / (2 * 0.78 * 1.2**2)) ** -0.78
    noise = 0.18**2 * np.exp(-(r**2) / (2 * (1.6 / 12) ** 2))
    return trend + season + medium + noise


def co2_sklearn():
    """The same kernel in scikit-learn."""
    k = (skk.ConstantKernel(66.0**2) * skk.RBF(67.0)
         + skk.ConstantKernel(2.4**2) * skk.RBF(90.0)
         * skk.ExpSineSquared(1.3, 1.0)
         + skk.ConstantKernel(0.66**2) * skk.RationalQuadratic(1.2, 0.78)
         + skk.ConstantKernel(0.18**2) * skk.RBF(1.6 / 12))
    return GaussianProcessRegressor(k, alpha=CO2_NOISE**2, optimizer=None)


def co2(lib):
    path = fetch("https://gml.noaa.gov/webdata/ccgg/trends/co2/"
                 "co2_weekly_mlo.txt", "co2_weekly_mlo.txt")
    d = np.loadtxt(path, comments="#")
    t, ppm = d[:, 3], d[:, 4]
    t, ppm = t[ppm > 0], ppm[ppm > 0]  # -999.99: missing week
    print(f"\n== CO2 at Mauna Loa: {len(t)} weeks, {t[0]:.1f} to "
          f"{t[-1]:.1f} ==")
    errs = []

    def predict(train, test):
        mu = ppm[train].mean()
        K = co2_kernel(t[train], t[train]) + CO2_NOISE**2 * np.eye(
            train.sum())
        Ks = co2_kernel(t[train], test)
        Kss = co2_kernel(test, test)
        mean, cov = lib.gp(K, Ks, Kss, ppm[train] - mu)
        errs.append(difference((mean, cov),
                               reference(K, Ks, Kss, ppm[train] - mu)))
        if GaussianProcessRegressor:
            g = co2_sklearn().fit(t[train, None], ppm[train] - mu)
            errs.append(difference((mean, cov),
                                   g.predict(test[:, None], return_cov=True)))
        return mean[:, 0] + mu, np.sqrt(np.diag(cov) + CO2_NOISE**2)

    # back test: trained until 2016, the ten years after predicted
    train = t < 2016
    back, back_sd = predict(train, t[~train])
    e = back - ppm[~train]
    print(f"back test, trained until 2016 ({train.sum()} weeks), "
          f"{(~train).sum()} weeks predicted:")
    print(f"  RMSE (root mean square error) {np.sqrt(np.mean(e**2)):.2f} "
          f"ppm, mean error {e.mean():+.2f} ppm, within 2 sd: "
          f"{100 * np.mean(np.abs(e) < 2 * back_sd):.0f} %")
    print("  (the kernel of 2006 underestimates the accelerating trend)")

    # forecast from all data
    future = np.arange(np.ceil(t[-1] * 12) / 12, 2036, 1 / 12)
    mean, sd = predict(np.ones_like(t, dtype=bool), future)
    print(f"forecast from all {len(t)} weeks, {len(future)} months:")
    print("  year    ppm   +-2 sd")
    for y in range(int(np.ceil(future[0])), 2036, 3):
        i = np.argmin(np.abs(future - y))
        print(f"  {y}  {mean[i]:6.1f}  {2 * sd[i]:5.1f}")
    print("relative difference to "
          + ("SciPy and scikit-learn" if GaussianProcessRegressor
             else "NumPy") + f": {max(errs):.1e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return max(errs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, ppm, lw=0.6, color="0.3", label="measured (weekly)")
    for x, m, s, c, lab in ((t[~train], back, back_sd, "tab:orange",
                             "back test: trained until 2016"),
                            (future, mean, sd, "tab:blue", "forecast")):
        ax.plot(x, m, color=c, lw=1, label=lab)
        ax.fill_between(x, m - 2 * s, m + 2 * s, color=c, alpha=0.25,
                        lw=0)
    ax.set_xlim(1990, 2036)
    ax.set_ylim(340, 470)
    ax.set_xlabel("year")
    ax.set_ylabel("CO$_2$ [ppm]")
    ax.set_title("GP regression with ulm.lam: CO$_2$ at Mauna Loa "
                 "(mean $\\pm$ 2 sd)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "co2.png", dpi=150)
    print(f"plot: {OUT / 'co2.png'}")
    return max(errs)


# ---------------------------------------------------------------------------
# Bike sharing
# ---------------------------------------------------------------------------


def bike_data():
    path = fetch("https://archive.ics.uci.edu/static/public/275/"
                 "bike+sharing+dataset.zip", "bike_sharing.zip")
    with zipfile.ZipFile(path) as z, z.open("hour.csv") as f:
        names = f.readline().decode().strip().split(",")
        d = np.loadtxt(f, delimiter=",", usecols=range(2, len(names)))
    c = {n: d[:, i - 2] for i, n in enumerate(names) if i >= 2}

    def cyc(v, p):
        return [np.sin(2 * np.pi * v / p), np.cos(2 * np.pi * v / p)]

    X = np.column_stack([*cyc(c["hr"], 24), *cyc(c["mnth"], 12),
                         *cyc(c["weekday"], 7), c["yr"], c["workingday"],
                         c["holiday"], c["weathersit"], c["temp"],
                         c["atemp"], c["hum"], c["windspeed"]])
    X = (X - X.mean(0)) / X.std(0)
    y = np.log1p(c["cnt"])
    return X, (y - y.mean()) / y.std()


def rbf(A, B, var):
    """var * exp(-|a - b|^2 / 2), A and B scaled by the length scales."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return var * np.exp(-0.5 * np.maximum(d2, 0))


def bike_hyperparameters(X, y):
    """Length scale per feature, signal and noise variance: maximum of the
    log marginal likelihood (scikit-learn), or a coarse grid (NumPy)."""
    if GaussianProcessRegressor:
        k = (skk.ConstantKernel(1.0) * skk.RBF(np.full(X.shape[1], 3.0))
             + skk.WhiteKernel(0.05))
        g = GaussianProcessRegressor(k, n_restarts_optimizer=0,
                                     random_state=0).fit(X, y)
        p = g.kernel_.get_params()
        return (p["k1__k2__length_scale"], p["k1__k1__constant_value"],
                p["k2__noise_level"], "scikit-learn")
    best_lml, best_p = -np.inf, None
    for ell in (1.0, 2.0, 3.0, 4.0, 6.0):
        for s2 in (0.0025, 0.01, 0.04, 0.16):
            K = rbf(X / ell, X / ell, 1.0) + s2 * np.eye(len(y))
            L = np.linalg.cholesky(K)
            a = np.linalg.solve(L.T, np.linalg.solve(L, y))
            lml = -0.5 * y @ a - np.log(np.diag(L)).sum()
            if lml > best_lml:
                best_lml, best_p = lml, (np.full(X.shape[1], ell), 1.0, s2)
    return (*best_p, "NumPy grid")


def bike(lib, sizes):
    X, y = bike_data()
    rng = np.random.default_rng(1)
    p = rng.permutation(len(y))
    test, pool = p[:1000], p[1000:]
    print(f"\n== Bike sharing: {len(y)} hours, {X.shape[1]} features, "
          f"{len(test)} test hours ==")
    sub = pool[:1500]
    t0 = time.perf_counter()
    ell, var, s2, how = bike_hyperparameters(X[sub], y[sub])
    print(f"hyperparameters by {how} on {len(sub)} hours "
          f"({time.perf_counter() - t0:.1f} s): signal variance {var:.2f}, "
          f"noise variance {s2:.3f}")
    Xs = X / ell
    rows = []
    for n in sizes:
        tr = pool[:n]
        t0 = time.perf_counter()
        K = rbf(Xs[tr], Xs[tr], var) + s2 * np.eye(n)
        Ks = rbf(Xs[tr], Xs[test], var)
        Kss = rbf(Xs[test], Xs[test], var)
        t_kernel = time.perf_counter() - t0
        yt = y[tr].copy()
        reps = 3 if n <= 4000 else 1
        t_lam, got = best(lambda: lib.gp(K, Ks, Kss, yt), reps)
        t_ref, ref = best(lambda: reference(K, Ks, Kss, yt), reps)
        t_lit = (best(lambda: numpy_literal(K, Ks, Kss, yt), 1)[0]
                 if n <= 8000 else float("nan"))
        t_sk = float("nan")
        if GaussianProcessRegressor:
            k = (skk.ConstantKernel(var) * skk.RBF(ell))
            g = GaussianProcessRegressor(k, alpha=s2, optimizer=None)
            t_sk, _ = best(lambda: g.fit(X[tr], yt).predict(
                X[test], return_cov=True), reps)
        rmse = np.sqrt(np.mean((got[0][:, 0] - y[test]) ** 2))
        rows.append((n, rmse, t_kernel, t_lam, t_ref, t_lit, t_sk,
                     difference(got, ref)))

    print("RMSE in standard deviations of log(1 + rentals); times in s,"
          " (x): relative to ulm.lam;")
    print("kernel: the kernel matrices in NumPy, the others without them "
          "(scikit-learn: with them)")
    print(f"      n   RMSE  kernel  ulm.lam  "
          f"{'SciPy' if cholesky else 'NumPy'} by hand    NumPy literal"
          "      scikit-learn  difference")

    def f(x, ref):
        return ("        -       " if x != x
                else f"{x:7.3f} ({x / ref:4.1f}x)")

    for n, rmse, tk, tl, tr_, tt, ts, err in rows:
        print(f"  {n:5d}  {rmse:5.3f}  {tk:6.3f}  {tl:7.3f}  {f(tr_, tl)}"
              f"  {f(tt, tl)}  {f(ts, tl + tk)}  {err:.1e}")
    return max(r[-1] for r in rows)


def main():
    full = "--full" in sys.argv
    t0 = time.perf_counter()
    lib = lam.compile(gp, name="gp_py")
    print(f"compiled in {time.perf_counter() - t0:.1f} s: {lib.path}")
    print("generated C++ (lib.gp.cpp):\n")
    print(lib.gp.cpp)
    err = co2(lib)
    sizes = [1000, 2000, 4000, 8000] + ([16000] if full else [])
    err = max(err, bike(lib, sizes))
    ok = err < 1e-8
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
