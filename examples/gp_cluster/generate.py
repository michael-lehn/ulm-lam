"""Generate examples/gp_cluster/generated/gp.cpp (calling ulmBLAS) and
gp_cblas.cpp (the same calls on CBLAS/LAPACK, backend="cblas"): the GP
prediction of python/examples/gp.py, with K as inout (it may be
overwritten by its Cholesky factor: no n x n copy, at n = 100000 that is
80 GB)."""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

from ulm import lam  # noqa: E402
from ulm.lam import InOut, Matrix, Vector  # noqa: E402


@lam.function
def gp(K: InOut(Matrix("n", "n", "spd")), Ks: Matrix("n", "m"),
       Kss: Matrix("m", "m", "symmetric"), y: Vector("n")):
    mean = Ks.T @ K.inv @ y
    cov = Kss - Ks.T @ K.inv @ Ks
    return mean, cov


(HERE / "generated").mkdir(exist_ok=True)
for backend, name in (("ulmblas", "gp"), ("cblas", "gp_cblas")):
    lam.compile(gp, name=f"gp_cluster_{backend}", backend=backend)
    shutil.copy(HERE.parents[1] / "build" / "python" / f"gp_cluster_{backend}"
                / f"gp_cluster_{backend}.cpp",
                HERE / "generated" / f"{name}.cpp")
    print(f"generated/{name}.cpp")
