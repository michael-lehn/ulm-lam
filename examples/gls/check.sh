#!/usr/bin/env bash
# The GLS derivation found by ulm-lam-select-kernels from 02-normal.mlir:
# M and S = W^T W are factorised once each (Cholesky, no LU), W = L^-1 X is
# computed once, S is formed from it (common subexpression up to
# transposition) and the right-hand side reuses it (W^T z), and no inverse
# is formed.  As in LAPACK, gls may overwrite its matrices: it works in M,
# X and y themselves, without copies; the wrapper gls_keep that ulm.lam
# adds copies exactly the three inputs that kernels overwrite (M, X, y).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/gls}"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/02-normal.mlir" "$OUT/gls"
SEL="$OUT/gls.selected.mlir" BUF="$OUT/gls.bufferized.mlir"
n() { grep -c "$1" "$2" || true; }
# the function @$1 of the IR in $2 (to count in one function only)
fn() { awk -v f="$1" '$0 ~ "func.func @" f "\\(" {p = 1} p {print} p && /^  }$/ {exit}' "$2"; }
fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
check "two Cholesky factorisations, no LU" '[ "$(n ulm_lamk.potrf "$SEL")" = 2 ] && [ "$(n ulm_lamk.getrf "$SEL")" = 0 ]'
check "one trsm (W = L^-1 X)" '[ "$(n ulm_lamk.trsm "$SEL")" = 1 ]'
w=$(grep -oE "ulm_lamk.syrk uplo\(lower\) trans\(trans\).*ins\(%[0-9]+" "$SEL" | grep -oE "%[0-9]+$" || true)
check "S = W^T W by syrk (one triangle) from W = L^-1 X ($w)" '[ -n "$w" ] && grep -qE "^ *$w = ulm_lamk.trsm" "$SEL"'
check "W reused for W^T z: three trsv (one with L, two with K)" '[ "$(n ulm_lamk.trsv "$SEL")" = 3 ]'
check "no ulm_lam op left" '! grep -q "ulm.lam\.\(mul\|add\|transpose\|inverse\)" "$SEL"'
check "gls: no copy (it works in M, X, y)" '[ "$(fn gls "$BUF" | grep -c memref.copy)" = 0 ]'
check "gls_keep: three copies (M, X, y), then gls" '[ "$(fn gls_keep "$BUF" | grep -c memref.copy)" = 3 ] && fn gls_keep "$BUF" | grep -q "call @gls("'
check "b written into the output buffer (workspace last)" 'grep -qE "@gls\(.*%arg3: memref<200x1xf64> \{bufferize.result\}, %arg4: memref<\?xi8>[^,]*, %arg5: memref<1xi64>[^)]*\)" "$BUF"'

# gls_inout.mlir: X is read only ({ulm_lam.in}), M and y may be
# overwritten: only X is copied (for the in-place W = L^-1 X)
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/gls_inout.mlir" "$OUT/gls_inout"
BUF="$OUT/gls_inout.bufferized.mlir"
check "inout: one copy (X), none of M and y" '[ "$(fn gls_inout "$BUF" | grep -c memref.copy)" = 1 ]'
check "inout: potrf works on the caller's M" 'grep -q "ulm_lamk.potrf uplo(lower) outs(%arg1 " "$BUF"'
check "inout: the interface comment says so" 'grep -q "v2  double\[2000\]\[2000\]  inout  may be overwritten" "$OUT/gls_inout.cpp"'
exit $fail
