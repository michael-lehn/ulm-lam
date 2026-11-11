#!/usr/bin/env bash
# accumulate.mlir (ulm_lam ops) to C++ with ulmBLAS by passes alone; compile,
# link with the C driver (foo and the naive reference), run, verify.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/loop}"
. "$ROOT/scripts/ulmblas-env.sh"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/accumulate.mlir" "$OUT/accumulate" "$HERE/generated"
"$CXX" "${ULMFLAGS[@]}" -Wall -Wextra -pedantic -c "$OUT/accumulate.cpp" \
	-o "$OUT/accumulate.o"
"$CC" -std=c11 -O2 -c "$HERE/driver.c" -o "$OUT/driver.o"
"$CXX" "$OUT/accumulate.o" "$OUT/driver.o" -o "$OUT/accumulate"
"$OUT/accumulate"
