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
| `driver.cc` | kernel matrices, `gp`, MKL by hand; `GP_HAND_FIRST=1` swaps the order, `GP_PROF=1` times the MKL calls, `GP_SERIAL_TOUCH=1` zeroes arrays with the main thread |
| `build.sh`, `run.sbatch` | build on the login node (GCC 14.2, `-march=native`, OpenMP, MKL 2025.0), Slurm job on an exclusive Zen 4 node |

## Results (2026-11-12)

One exclusive node of the `dev_cpu` partition: 2x AMD EPYC 9454 (Zen 4),
96 cores, 96 OpenMP threads (`OMP_PLACES=cores`, `OMP_PROC_BIND=spread`).
Times in seconds, without the kernel matrices.  ulmBLAS `master`
(b16a89aa), with its default AVX/FMA micro kernel for `-march=native`.

| n | RMSE [m] | ulm.lam on ulmBLAS | ulm.lam on MKL | by hand on MKL |
|---:|---:|---:|---:|---:|
| 4 000 | 11.03 | 0.44 | 0.17 | 0.16 |
| 12 500 | 6.85 | 2.48 | 1.34 | 1.46 |
| 25 000 | 4.83 | 13.66 | 5.71 | 5.85 |
| 50 000 | 3.17 | 106.5 | 26.15 | 27.84 |
| 100 000 | 1.95 | 590.7 | 153.7 | 150.7 |

ulm.lam ran first in each process; the arrays are first written by the
threads (see below).  The kernel matrices take 1.1 s at n = 100 000.

The RMSE (root mean square error) of the predicted altitude falls from
15.5 m (n = 1000) to 1.95 m: exact GP regression profits from all the data.
ulm.lam and MKL agree to 1e-13; ulm.lam on MKL and by hand on MKL give
identical results (the same calls).

**The mapping (ulm.lam on MKL vs by hand on MKL)** is as fast as the
hand-written version (within 6% either way).  The first measurements
used a driver that zeroed its arrays with `std::vector`, i.e. with the
main thread: all pages lay on one of the 8 NUMA domains, and the
kernel's automatic NUMA balancing (`numa_balancing` = 1) migrated them
during the first heavy run, so whichever variant ran first paid about
25%.  `driver.cc` now leaves large arrays uninitialised and writes them first
with the threads (kernel matrices, workspace, copies); then the order no
longer matters (n = 50 000, MKL):

| first touch | ulm.lam first: ulm.lam / by hand | by hand first: ulm.lam / by hand |
|---|---:|---:|
| main thread (`GP_SERIAL_TOUCH=1`) | 31.66 / 27.28 | 27.82 / 30.02 |
| the threads (default) | **26.00 / 27.35** | **26.37 / 27.42** |
| main thread, `numactl --interleave=all` | 25.29 / 27.85 | |

and the kernel matrices take 0.27 s instead of 2.3 s.  The generated code
copies large arrays in parallel (`ulm_lam_copy`) for the same reason.

**ulmBLAS (ulm.lam on ulmBLAS vs on MKL)** is 3.8x slower at n = 100 000.
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
