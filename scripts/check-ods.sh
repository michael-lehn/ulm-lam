#!/usr/bin/env bash
# Run every mlir-tblgen backend the dialect needs; fails on the first error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm 2>/dev/null || echo /usr/local)}"
TBLGEN="$LLVM/bin/mlir-tblgen"
OUT="${OUT:-$ROOT/build/gen}"
mkdir -p "$OUT"
run() { # file backend [extra...]
  local f=$1 b=$2; shift 2
  "$TBLGEN" "$b" "$@" -I "$ROOT/include" -I "$LLVM/include" \
      "$ROOT/include/ulm_lam/$f" -o "$OUT/${f%.td}$b.inc"
  echo "ok   $f $b $*"
}
run UlmLamBase.td        -gen-dialect-decls -dialect=ulm_lam
run UlmLamBase.td        -gen-dialect-decls -dialect=ulm_lamk
run UlmLamAttrs.td       -gen-enum-decls
run UlmLamAttrs.td       -gen-enum-defs
run UlmLamAttrs.td       -gen-attrdef-decls -attrdefs-dialect=ulm_lam
run UlmLamAttrs.td       -gen-attrdef-defs  -attrdefs-dialect=ulm_lam
run UlmLamAttrs.td       -gen-attrdef-decls -attrdefs-dialect=ulm_lamk
run UlmLamAttrs.td       -gen-attrdef-defs  -attrdefs-dialect=ulm_lamk
run UlmLamInterfaces.td  -gen-op-interface-decls
run UlmLamInterfaces.td  -gen-op-interface-defs
run UlmLamOps.td         -gen-op-decls
run UlmLamOps.td         -gen-op-defs
run UlmLamOps.td         -gen-op-doc -dialect=ulm_lam
run UlmLamKernelOps.td   -gen-op-decls
run UlmLamKernelOps.td   -gen-op-defs
run UlmLamKernelOps.td   -gen-op-doc -dialect=ulm_lamk
run UlmLamPasses.td      -gen-pass-decls
"$LLVM/bin/mlir-tblgen" --print-records -I "$ROOT/include" \
    "$ROOT/include/ulm_lam/UlmLamKernels.td" > "$OUT/UlmLamKernels.records"
echo "ok   UlmLamKernels.td (records)"
