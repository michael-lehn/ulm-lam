#!/usr/bin/env bash
# Formula (1) in a loop: everything that does not depend on y_i runs once,
# before the loop.  The loop body of the selected kernels has no
# factorisation, no gemm and no trsm, one solve with the factor of R, a
# gemv with W = L^-1 H (reused from S = W^T W) and two solves with the
# factor of S; after bufferization it has no allocation or copy.
#
# adaptive.mlir (R updated in place in every step): the update is a
# ulm_lamk.call that updates the loop-carried R in place, each step factorises
# R_i and S (2 potrf) and computes W once (1 trsm), and the bufferized loop
# has no allocation and exactly two copies (R_i for the in-place potrf, H
# for the in-place trsm).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
OUT="${OUT:-$ROOT/build/kalman}"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/estimate.mlir" "$OUT/estimate"
SEL="$OUT/estimate.selected.mlir" BUF="$OUT/estimate.bufferized.mlir"
# ops inside the (first) scf.for of FN, by brace depth
loop_body() { awk -v fn="${2:-}" 'fn != "" && !s && index($0, "func.func @" fn) {s=1} (fn == "" || s) && !f && /scf.for/ {f=1; d=0} f { o=gsub(/\{/, "{"); c=gsub(/\}/, "}"); if (p) print; p=1; d+=o-c; if (d<=0) exit }' "$1"; }
count() { loop_body "$1" | grep -c "$2" || true; }
fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
check "no factorisation in the loop" '[ "$(count "$SEL" "ulm_lamk.potrf\|ulm_lamk.getrf")" = 0 ]'
check "no gemm or trsm in the loop" '[ "$(count "$SEL" "ulm_lamk.gemm\|ulm_lamk.trsm")" = 0 ]'
# the loop body has two variants, chosen before the loop from the trip
# count: one gemv with the operator P = S^-1 H^T R^-1 formed before the
# loop, or 3 trsv and 1 gemv (few steps)
check "loop body: 1 gemv, or 3 trsv and 1 gemv" '[ "$(count "$SEL" ulm_lamk.trsv)" = 3 ] && [ "$(count "$SEL" ulm_lamk.gemv)" = 2 ]'
check "operator P^T = L^-T W S^-1 before the loop, if the loop runs often enough" '[ "$(sed -n "/scf.if/,/scf.for/p" "$SEL" | grep -c ulm_lamk.trsm)" = 3 ]'
check "two Cholesky factorisations before the loop" '[ "$(grep -c ulm_lamk.potrf "$SEL")" = 2 ]'
check "no allocation or copy in the loop" '[ "$(count "$BUF" "memref.alloc\|memref.copy")" = 0 ]'
check "observe(i, y) by reference" 'grep -q "func.func private @observe(index, memref<2000x1xf64>)$" "$BUF"'

"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/adaptive.mlir" "$OUT/adaptive"
SEL="$OUT/adaptive.selected.mlir" BUF="$OUT/adaptive.bufferized.mlir"
check "adaptive: no ulm_lam op left" '! grep -q "ulm.lam\.\(mul\|add\|transpose\|inverse\)" "$SEL"'
check "adaptive: in-place update of R (ulm_lamk.call)" 'loop_body "$BUF" | grep -qE "ulm_lamk.call @update_covariance\(%arg[0-9]+\) outs\(%(alloc|view)[_0-9]*\)"'
check "adaptive: 2 potrf and 1 trsm per step" '[ "$(count "$SEL" ulm_lamk.potrf)" = 2 ] && [ "$(count "$SEL" ulm_lamk.trsm)" = 1 ]'
check "adaptive: no allocation in the loop" '[ "$(count "$BUF" memref.alloc)" = 0 ]'
check "adaptive: two copies per step (R_i, H)" '[ "$(count "$BUF" memref.copy)" = 2 ]'
exit $fail
