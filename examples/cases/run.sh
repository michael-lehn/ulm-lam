#!/usr/bin/env bash
# Kernel selection cases (cases.mlir) to C++ with ulmBLAS; compile, run,
# compare with naive evaluations (driver.cc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/cases}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/cases.mlir" "$OUT/cases" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/cases.cpp" -o "$OUT/cases.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/cases.o" "$HERE/driver.cc" -o "$OUT/cases" "${ULMLIBS[@]}"
"$OUT/cases"
