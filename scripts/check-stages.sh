#!/usr/bin/env bash
# The hand-written GLS stages against the implemented dialects:
#   * stages 1-4 parse and verify (types, inferred properties, DPS rules);
#   * stage 3 through upstream One-Shot Bufferize with the ulm_lamk
#     bufferization models gives exactly the copies stage 4 predicts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
OUT="${OUT:-$ROOT/build/stages}"
mkdir -p "$OUT"
PLUGIN="$("$ROOT/scripts/build-plugin.sh")"
opt() { "$LLVM/bin/mlir-opt" --load-dialect-plugin="$PLUGIN" --load-pass-plugin="$PLUGIN" "$@"; }
fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
for f in "$ROOT"/examples/gls/0[1-4]-*.mlir; do
	check "$(basename "$f") parses and verifies" 'opt "$f" -o /dev/null'
done
sed 's/func.func @gls(%X: !X, %M: !M, %y: !y) -> !b {/func.func @gls(%X: !X {bufferization.writable = false}, %M: !M {bufferization.writable = false}, %y: !y {bufferization.writable = false}) -> !b {/' \
	"$ROOT/examples/gls/03-kernels.mlir" >"$OUT/03-readonly.mlir"
opt "$OUT/03-readonly.mlir" -o "$OUT/04-generated.mlir" --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}, buffer-deallocation-pipeline, canonicalize)'
copies=$(grep -c "memref.copy" "$OUT/04-generated.mlir" || true)
check "stage 3 bufferizes with 3 copies (M, X, y), as stage 4 predicts (got $copies)" '[ "$copies" = 3 ]'
check "gemv and both trsv share one buffer" '[ "$(grep -E "ulm_lamk.(gemv|trsv).*outs\(%alloc_4" "$OUT/04-generated.mlir" | wc -l | tr -d " ")" = 3 ]'
exit $fail
