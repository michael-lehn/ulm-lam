#!/usr/bin/env bash
# Build ulm-lam-opt (tools/ulm-lam-opt.cpp): mlir-opt with the ulm.lam
# dialects and passes, linked against the plugin library (built first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
PLUGIN="$("$ROOT/scripts/build-plugin.sh")"
OUT="$ROOT/build/bin/ulm-lam-opt"
mkdir -p "$(dirname "$OUT")"
if [ -e "$OUT" ] && [ "$OUT" -nt "$PLUGIN" ] && [ "$OUT" -nt "$ROOT/tools/ulm-lam-opt.cpp" ]; then
	echo "$OUT"
	exit 0
fi
"$LLVM/bin/clang++" -std=c++17 -O2 -fno-rtti \
	-I "$LLVM/include" -I "$ROOT/lib" -I "$ROOT/build/gen-plugin" \
	"$ROOT/tools/ulm-lam-opt.cpp" "$PLUGIN" \
	-L "$LLVM/lib" -lMLIR -lLLVM \
	-Wl,-rpath,"$LLVM/lib" -Wl,-rpath,"$(dirname "$PLUGIN")" \
	-o "$OUT"
echo "$OUT"
