#!/usr/bin/env bash
# ulm-lam-compile.sh IN.mlir OUTPREFIX [SHOWDIR]
#
# The ulm.lam pipeline from ulm_lam ops to C++ calling ulmBLAS (or, with
# BACKEND=cblas, CBLAS and LAPACK).  Writes
#   OUTPREFIX.selected.mlir    after kernel selection (ulm_lamk on tensors)
#   OUTPREFIX.bufferized.mlir  after bufferization (ulm_lamk on memrefs)
#   OUTPREFIX.emitc.mlir       after convert-ulm-lamk-to-ulmblas
#   OUTPREFIX.cpp              C++17, formatted with clang-format
# and copies the C++ into SHOWDIR (the examples keep their generated code
# in examples/*/generated/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
IN=$1 OUT=$2 SHOW=${3:-}
mkdir -p "$(dirname "$OUT")"
PLUGIN="$("$ROOT/scripts/build-plugin.sh")"
opt() {
	"$LLVM/bin/mlir-opt" --load-dialect-plugin="$PLUGIN" \
		--load-pass-plugin="$PLUGIN" "$@"
}
# LAPACK's L D L^T and QR where the backend has them (CBLAS/LAPACK);
# ULM_LAM_SELECT_OPTIONS: further options of ulm-lam-select-kernels
# (e.g. expand-factors=false)
LAPACK=false
[ "${BACKEND:-ulmblas}" = cblas ] && LAPACK=true
SELECT="func.func(ulm-lam-normalize, ulm-lam-infer-properties${LICM-, loop-invariant-code-motion}, ulm-lam-select-kernels{lapack=$LAPACK ${ULM_LAM_SELECT_OPTIONS:-}}), ulm-lam-strip-properties, canonicalize"
BUFFERIZE='ulm-lam-results-to-out-params, eliminate-empty-tensors,
	one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
	buffer-results-to-out-params{hoist-static-allocs modify-public-functions add-result-attr},
	func.func(buffer-loop-hoisting), buffer-deallocation-pipeline, canonicalize,
	ulm-lam-workspace, canonicalize, cse, ulm-lam-info, canonicalize'
# BACKEND=cblas: the same calls on CBLAS/LAPACK (include/ulm_lam/cblas_backend.h)
LOWER="convert-ulm-lamk-to-ulmblas{backend=${BACKEND:-ulmblas}}, reconcile-unrealized-casts, form-expressions"

opt "$IN" --pass-pipeline="builtin.module($SELECT)" -o "$OUT.selected.mlir"
opt "$OUT.selected.mlir" --pass-pipeline="builtin.module($BUFFERIZE)" \
	-o "$OUT.bufferized.mlir"
opt "$OUT.bufferized.mlir" --pass-pipeline="builtin.module($LOWER)" \
	-o "$OUT.emitc.mlir"
"$LLVM/bin/mlir-translate" --mlir-to-cpp "$OUT.emitc.mlir" -o "$OUT.cpp"
# Format with the clang-format style found from the output location
# (the project's or the user's .clang-format).
"$LLVM/bin/clang-format" -i "$OUT.cpp"
if [ -n "$SHOW" ]; then
	mkdir -p "$SHOW"
	cp "$OUT.cpp" "$SHOW/$(basename "$OUT").cpp"
fi
