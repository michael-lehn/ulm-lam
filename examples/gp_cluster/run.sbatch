#!/usr/bin/env bash
# GP regression on the 3D Road Network data, n up to 100000, on one
# exclusive Zen 4 node (2x AMD EPYC 9454, 96 cores, 384 GB).
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=360G
#SBATCH --time=01:30:00
#SBATCH --job-name=ulm-lam-gp
#SBATCH --output=gp-%j.out
set -euo pipefail
module load compiler/gnu/14.2 numlib/mkl/2025.0
export OMP_NUM_THREADS=96 MKL_NUM_THREADS=96
export OMP_PLACES=cores OMP_PROC_BIND=spread
lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"
./gpdrv road.bin hyper.txt 2000 ${SIZES:-12500 25000 50000 100000}
