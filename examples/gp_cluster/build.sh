#!/usr/bin/env bash
# Build the driver on bwUniCluster 3.0 (login node: compiling one program
# is allowed there; -march=native on the login node is Zen 4, the CPU of
# the cpu partition).  ulmBLAS (header only) selects its micro kernels
# from the target flags.  -fopenmp: potrf and trtrs of ulmBLAS run as
# OpenMP tasks.  MKL (Intel Math Kernel Library) with GNU OpenMP threads,
# for the comparison.
set -euo pipefail
cd "$(dirname "$0")"
module load compiler/gnu/14.2 numlib/mkl/2025.0
g++ -std=c++17 -O3 -march=native -fopenmp -DWITH_MKL -isystem . \
	-I"$MKLROOT/include" driver.cc gp.cpp -o gpdrv \
	-L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
	-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
	-lgomp -lpthread -lm -ldl
echo "built $(pwd)/gpdrv"
