# Sourced by the run scripts: where ulmBLAS is and how to compile with it.
#   ULMBLAS_DIR  top-level directory of ulmBLAS (contains ulmblas/ulmblas.h)
#   ARCHFLAGS    flags for the optimised micro kernels (see ulmBLAS
#                BUILDING.md), e.g. -mcpu=apple-m4 or '-mavx -mfma'
ULMBLAS_DIR="${ULMBLAS_DIR:-$HOME/git/ulmBLAS}"
ARCHFLAGS="${ARCHFLAGS:--mcpu=native}"
if [ ! -f "$ULMBLAS_DIR/ulmblas/ulmblas.h" ]; then
	echo "ulmBLAS not found: set ULMBLAS_DIR (now $ULMBLAS_DIR)" >&2
	exit 1
fi
CXX="${CXX:-c++}"
CC="${CC:-cc}"
# shellcheck disable=SC2206
# -isystem: warnings are reported for the generated code, not for ulmBLAS
ULMFLAGS=(-std=c++17 -O3 $ARCHFLAGS -isystem "$ULMBLAS_DIR")
# OpenMP (OPENMP=0: without): ulmBLAS runs potrf, getrf and the solves
# (trtrs, getrs, potrs) as OpenMP tasks.  Apple clang needs Homebrew's
# libomp.  ULMLIBS: for the link commands.
ULMLIBS=()
if [ "${OPENMP:-1}" = 1 ]; then
	if [ "$(uname)" = Darwin ]; then
		OMPDIR="$(brew --prefix libomp)"
		ULMFLAGS+=(-Xpreprocessor -fopenmp -isystem "$OMPDIR/include")
		ULMLIBS=(-L"$OMPDIR/lib" -Wl,-rpath,"$OMPDIR/lib" -lomp)
	else
		ULMFLAGS+=(-fopenmp)
		ULMLIBS=(-fopenmp)
	fi
fi
