"""Data and hyperparameters for the large GP run (examples/gp_cluster).

3D Road Network (North Jutland, Denmark), UCI Machine Learning Repository
(Kaul, Yang, Jensen, Building accurate 3D spatial networks to enable next
generation intelligent transportation systems, MDM 2013): 434 874 points
(longitude, latitude, altitude).  The GP predicts the altitude from the
position, a kriging problem.

Writes build/gp_cluster/road.bin: float64 rows (x [km], y [km], altitude
standardised) in random order, the first M rows are the test points; and
hyper.txt: length scale [km], signal variance, noise variance of a
Matern 3/2 kernel, fitted by scikit-learn on the training points of the
largest run that fall into a window (so the density is that of n = NMAX).

    python3 examples/gp_cluster/prepare.py
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as skk

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "build" / "data"
OUT = ROOT / "build" / "gp_cluster"
M, NMAX = 2000, 100000

zp = DATA / "3droad.zip"
if not zp.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/246/"
        "3d+road+network+north+jutland+denmark.zip", zp)
with zipfile.ZipFile(zp) as z, z.open("3D_spatial_network.txt") as f:
    d = np.loadtxt(f, delimiter=",", usecols=(1, 2, 3))
lon, lat, alt = d.T
lat0 = np.deg2rad(lat.mean())
x = (lon - lon.mean()) * 111.32 * np.cos(lat0)
y = (lat - lat.mean()) * 110.57
z = (alt - alt.mean()) / alt.std()
rng = np.random.default_rng(7)
p = rng.permutation(len(z))
P = np.column_stack([x, y, z])[p]
OUT.mkdir(parents=True, exist_ok=True)
P.astype("<f8").tofile(OUT / "road.bin")
print(f"{len(z)} points, area {np.ptp(x):.0f} x {np.ptp(y):.0f} km, "
      f"altitude {alt.mean():.1f} +- {alt.std():.1f} m")

train = P[M:M + NMAX]
cx, cy = np.median(train[:, 0]), np.median(train[:, 1])
for half in np.arange(2.0, 40.0, 0.5):
    w = train[(np.abs(train[:, 0] - cx) < half)
              & (np.abs(train[:, 1] - cy) < half)]
    if len(w) >= 2500:
        break
print(f"window {2 * half:.0f} x {2 * half:.0f} km: {len(w)} points")
k = (skk.ConstantKernel(1.0) * skk.Matern(1.0, nu=1.5)
     + skk.WhiteKernel(0.01))
g = GaussianProcessRegressor(k, n_restarts_optimizer=2, random_state=0)
g.fit(w[:, :2], w[:, 2] - w[:, 2].mean())
pr = g.kernel_.get_params()
ell, var, noise = (pr["k1__k2__length_scale"], pr["k1__k1__constant_value"],
                   pr["k2__noise_level"])
print(f"Matern 3/2: length scale {ell:.3f} km, signal variance {var:.3f}, "
      f"noise variance {noise:.5f}")
(OUT / "hyper.txt").write_text(f"{float(ell)!r} {float(var)!r} {float(noise)!r}\n")
