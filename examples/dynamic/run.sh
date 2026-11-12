#!/usr/bin/env bash
# Formula (1) with sizes known only at run time (gls.mlir): one compiled
# function for all sizes; compile, run, verify (driver.cc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/dynamic}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/gls.mlir" "$OUT/gls" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/gls.cpp" -o "$OUT/gls.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/gls.o" "$HERE/driver.cc" -o "$OUT/gls"
"$OUT/gls"
# D = A B C: the plan is chosen at run time (chain.mlir)
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/chain.mlir" "$OUT/chain" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/chain.cpp" -o "$OUT/chain.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/chain.o" "$HERE/driver_chain.cc" -o "$OUT/chain"
"$OUT/chain"
