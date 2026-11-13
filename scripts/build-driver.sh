#!/usr/bin/env bash
# Build ulm-lam (tools/ulm-lam.cpp): the whole compiler in one executable,
# ulm_lam IR to C++ or Julia, linked against the plugin library.
# STATIC=1: the ulm.lam sources and MLIR's static libraries linked in, no
# dependency on LLVM's shared libraries (as the binary package is built).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
PLUGIN="$("$ROOT/scripts/build-plugin.sh")"
GEN="$ROOT/build/gen-plugin"
if [ -n "${STATIC:-}" ]; then
	OUT="$ROOT/build/bin-static/ulm-lam"
	mkdir -p "$(dirname "$OUT")"
	libs=$("$LLVM/bin/llvm-config" --link-static --libs --system-libs 2>/dev/null)
	mlir=$(cd "$LLVM/lib" && ls libMLIR*.a | sed 's/^lib\(.*\)\.a$/-l\1/' | tr '\n' ' ')
	"$LLVM/bin/clang++" -std=c++17 -O2 -fno-rtti \
		-I "$LLVM/include" -I "$ROOT/lib" -I "$GEN" \
		"$ROOT/tools/ulm-lam.cpp" $(ls "$ROOT"/lib/*.cpp | grep -v Plugin.cpp) \
		-L "$LLVM/lib" -Wl,-search_paths_first $mlir $libs -o "$OUT"
	echo "$OUT"
	exit 0
fi
OUT="$ROOT/build/bin/ulm-lam"
mkdir -p "$(dirname "$OUT")"
if [ -e "$OUT" ] && [ "$OUT" -nt "$PLUGIN" ] && [ "$OUT" -nt "$ROOT/tools/ulm-lam.cpp" ]; then
	echo "$OUT"
	exit 0
fi
"$LLVM/bin/clang++" -std=c++17 -O2 -fno-rtti \
	-I "$LLVM/include" -I "$ROOT/lib" -I "$GEN" \
	"$ROOT/tools/ulm-lam.cpp" "$PLUGIN" \
	-L "$LLVM/lib" -lMLIR -lLLVM \
	-Wl,-rpath,"$LLVM/lib" -Wl,-rpath,"$(dirname "$PLUGIN")" \
	-o "$OUT"
echo "$OUT"
