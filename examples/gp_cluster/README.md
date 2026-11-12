# GP regression with n up to 100 000 on bwUniCluster 3.0

Exact GP (Gaussian process) regression on the 3D Road Network data (UCI,
University of California, Irvine, Machine Learning Repository; 434 874
points of the road network of North Jutland): the altitude predicted from
the position (kriging), m = 2000 test points, Matern 3/2 kernel.  The
prediction is the function `gp` of `python/examples/gp.py` (K as `inout`,
so the n x n matrix is factorised in place; at n = 100 000 it has 80 GB),
generated twice:

- `generated/gp.cpp`: calling ulmBLAS (default lowering);
- `generated/gp_cblas.cpp`: the same calls on CBLAS and LAPACK
  (`backend=cblas`, `include/ulm_lam/cblas_backend.h`), here linked
  against MKL (Intel Math Kernel Library).

`driver.cc` builds the kernel matrices (OpenMP), calls `gp`, and for
comparison does the same steps by hand with MKL, as SciPy or scikit-learn
on MKL would (`dpotrf`, `dpotrs`, `dgemv`, `dtrsm`, `dsyrk`).

| File | |
|---|---|
| `prepare.py` | downloads the data, writes `build/gp_cluster/road.bin` and fits the hyperparameters with scikit-learn (`hyper.txt`) |
| `generate.py` | generates `generated/gp.cpp` and `generated/gp_cblas.cpp` |
| `driver.cc` | kernel matrices, `gp`, MKL by hand; `GP_HAND_FIRST=1` swaps the order, `GP_PROF=1` times the MKL calls |
| `build.sh`, `run.sbatch` | build on the login node (GCC 14.2, `-march=native`, OpenMP, MKL 2025.0), Slurm job on an exclusive Zen 4 node |

## Results (2026-11-12)

One exclusive node of the `dev_cpu` partition: 2x AMD EPYC 9454 (Zen 4),
96 cores, 96 OpenMP threads (`OMP_PLACES=cores`, `OMP_PROC_BIND=spread`).
Times in seconds, without the kernel matrices.  ulmBLAS `master`
(b16a89aa), with its default AVX/FMA micro kernel for `-march=native`.

| n | RMSE [m] | ulm.lam on ulmBLAS | ulm.lam on MKL | by hand on MKL |
|---:|---:|---:|---:|---:|
| 4 000 | 11.03 | 0.48 | 0.32 | 0.26 |
| 12 500 | 6.85 | 3.50 | 2.22 | 2.18 |
| 25 000 | 4.83 | 19.8 | 7.49 | 7.57 |
| 50 000 | 3.17 | 137.0 | 26.59 | 26.60 |
| 100 000 | 1.95 | 579.0 | 168.0 | 154.8 |

The RMSE (root mean square error) of the predicted altitude falls from
15.5 m (n = 1000) to 1.95 m: exact GP regression profits from all the data.
ulm.lam and MKL agree to 1e-13; ulm.lam on MKL and by hand on MKL give
identical results (the same calls).

**The mapping (ulm.lam on MKL vs by hand on MKL)** is as fast as the
hand-written version.  Whichever of the two runs first in a process pays
about 25% (n = 50 000: by hand first 30.6 s, then ulm.lam 26.6 s; ulm.lam
first 33.7 s, then by hand 26.6 s); the table shows the second run where
both orders were measured.  At n = 100 000 only the order "ulm.lam first"
was run.

**ulmBLAS (ulm.lam on ulmBLAS vs on MKL)** is 3.4x slower at n = 100 000.
Per kernel (n = 25 000): `potrf` 14.5 s (ulmBLAS) vs 2.9 s (MKL), `trtrs`
3.2 s vs 4.1 s (`dtrsm`), `sylrk` 1.9 s vs 0.27 s.  Thread scaling at
n = 12 500:

| threads | ulmBLAS `potrf` | ulmBLAS `trtrs` | ulmBLAS `sylrk` | ulm.lam on ulmBLAS | by hand on MKL |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.76 | 5.49 | 0.96 | 18.28 | 22.63 |
| 12 | 2.32 | 1.02 | 0.97 | 4.39 | 2.56 |
| 48 | 2.10 | 0.90 | 0.97 | 4.04 | 1.32 |
| 96 | 1.79 | 0.85 | 0.97 | 3.68 | 1.46 |

Hints for ulmBLAS:
- single threaded, ulmBLAS is faster than MKL (18.3 s vs 22.6 s);
- `potrf` and `trtrs` (OpenMP tasks) scale up to about 12 threads (one
  NUMA (Non-Uniform Memory Access) domain, 12 cores per NPS4 node of the
  EPYC 9454) and hardly beyond;
- level-3 BLAS (`sylrk`, and `gemm` inside it) runs on one thread.

## Reproduce

```sh
python3 examples/gp_cluster/prepare.py      # build/gp_cluster/road.bin, hyper.txt
python3 examples/gp_cluster/generate.py     # generated/gp.cpp, gp_cblas.cpp
# copy ulmblas/, include/ulm_lam/cblas_backend.h (as ulm_lam/), driver.cc,
# generated/*.cpp, road.bin, hyper.txt, build.sh, run.sbatch to the cluster
./build.sh                                  # gpdrv (ulmBLAS); for gp_cblas.cpp see below
sbatch run.sbatch
```

The MKL variant is built like `build.sh` with `gp_cblas.cpp` instead of
`gp.cpp` and `-I .` for `ulm_lam/cblas_backend.h`.
