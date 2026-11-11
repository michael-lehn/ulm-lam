#!/usr/bin/env bash
# Formula (1) in a loop to C++ with ulmBLAS; compile, run, verify:
#   estimate.mlir  constant covariance, with and without
#                  loop-invariant-code-motion (driver.cc)
#   adaptive.mlir  covariance updated in place in every step
#                  (driver_adaptive.cc)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/kalman}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/estimate.mlir" "$OUT/estimate" "$HERE/generated"
LICM= "$ROOT/scripts/ulm-lam-compile.sh" "$HERE/estimate.mlir" "$OUT/estimate_nolicm"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/estimate.cpp" \
	-o "$OUT/estimate.o"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -Destimate=estimate_nolicm \
	-c "$OUT/estimate_nolicm.cpp" -o "$OUT/estimate_nolicm.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/estimate.o" "$OUT/estimate_nolicm.o" \
	"$HERE/driver.cc" -o "$OUT/estimate"
"$OUT/estimate"

"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/adaptive.mlir" "$OUT/adaptive" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/adaptive.cpp" \
	-o "$OUT/adaptive.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/adaptive.o" "$HERE/driver_adaptive.cc" \
	-o "$OUT/adaptive"
"$OUT/adaptive"
