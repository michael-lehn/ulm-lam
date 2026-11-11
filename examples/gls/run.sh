#!/usr/bin/env bash
# The GLS example from its ulm.lam normal form (02-normal.mlir), and the same
# with M and y {ulm_lam.inout} (gls_inout.mlir), to C++ with ulmBLAS by passes
# alone; compile, run, verify (driver.cc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/gls}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/02-normal.mlir" "$OUT/gls" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/gls.cpp" -o "$OUT/gls.o"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/gls_inout.mlir" "$OUT/gls_inout" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/gls_inout.cpp" \
	-o "$OUT/gls_inout.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/gls.o" "$OUT/gls_inout.o" "$HERE/driver.cc" \
	-o "$OUT/gls"
"$OUT/gls"
