#!/usr/bin/env bash
# Workspaces of functions that call functions with workspaces
# (compose.mlir) to C++ with ulmBLAS; compile, run, verify (driver.cc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/compose}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/compose.mlir" "$OUT/compose" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/compose.cpp" \
	-o "$OUT/compose.o"
"$CXX" "${ULMFLAGS[@]}" "$OUT/compose.o" "$HERE/driver.cc" -o "$OUT/compose" "${ULMLIBS[@]}"
"$OUT/compose"
