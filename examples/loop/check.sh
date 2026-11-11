#!/usr/bin/env bash
# X += A*B*C in a loop, from ulm_lam ops (scripts/ulm-lam-compile.sh).  Checks:
#   * kernel selection: no ulm_lam op left, two gemms, the second one
#     accumulates into the loop-carried X (beta = 1);
#   * memory: no allocation, copy or deallocation in the loop, foo(i, B)
#     takes B by reference and X is updated in place.  For comparison,
#     upstream bufferization alone allocates in every iteration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
OUT="${OUT:-$ROOT/build/loop}"
"$ROOT/scripts/ulm-lam-compile.sh" "$HERE/accumulate.mlir" "$OUT/accumulate"
PLUGIN="$("$ROOT/scripts/build-plugin.sh")"
"$LLVM/bin/mlir-opt" --load-dialect-plugin="$PLUGIN" --load-pass-plugin="$PLUGIN" \
	"$OUT/accumulate.selected.mlir" -o "$OUT/upstream.mlir" \
	--pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}, buffer-deallocation-pipeline, canonicalize)'
SEL="$OUT/accumulate.selected.mlir" BUF="$OUT/accumulate.bufferized.mlir"

# ops inside the (first) scf.for of FN, by brace depth
loop_body() { awk -v fn="${2:-}" 'fn != "" && !s && index($0, "func.func @" fn) {s=1} (fn == "" || s) && !f && /scf.for/ {f=1; d=0} f { o=gsub(/\{/, "{"); c=gsub(/\}/, "}"); if (p) print; p=1; d+=o-c; if (d<=0) exit }' "$1"; }
count() { loop_body "$1" accumulate | grep -c "$2" || true; }

fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
check "selection: no ulm_lam op left" '! grep -q "ulm.lam\.\(mul\|add\|transpose\|inverse\)" "$SEL"'
check "selection: two gemms in the loop" '[ "$(count "$SEL" ulm_lamk.gemm)" = 2 ]'
check "selection: X += T C with beta = 1" 'loop_body "$SEL" accumulate | grep "ulm_lamk.gemm" | tail -1 | grep -q "beta(%cst_0).*outs(%arg5"'
echo "upstream bufferization alone: $(count "$OUT/upstream.mlir" 'memref.alloc') allocations in the loop body"
check "ulm.lam: no allocation in the loop" '[ "$(count "$BUF" memref.alloc)" = 0 ]'
check "ulm.lam: no copy in the loop" '[ "$(count "$BUF" memref.copy)" = 0 ]'
check "ulm.lam: no deallocation in the loop" '[ "$(count "$BUF" dealloc)" = 0 ]'
check "ulm.lam: foo(i, B) by reference" 'grep -q "func.func private @foo(index, memref<500x500xf64>)$" "$BUF"'
check "ulm.lam: X in place, no result (workspace last)" 'grep -qE "@accumulate\(.*%arg3: index, %arg4: memref<\?xi8>[^,]*, %arg5: memref<1xi64>[^)]*\) \{$" "$BUF" && ! grep -q "memref.copy\|bufferization.clone" "$BUF"'
exit $fail
