#!/usr/bin/env bash
# Build the ulm.lam plugin for mlir-opt: the ulm.lam and ulm_lamk dialects
# (--load-dialect-plugin) and the ulm.lam passes (--load-pass-plugin), one
# shared library.  C++ declarations are generated from include/ulm_lam/*.td.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM="${LLVM_PREFIX:-$(brew --prefix llvm)}"
OUT="${OUT:-$ROOT/build/lib}"
GEN="$ROOT/build/gen-plugin"
mkdir -p "$OUT" "$GEN"
case "$(uname)" in Darwin) EXT=dylib ;; *) EXT=so ;; esac

tblgen() { # td backend out [extra...]
	local td=$1 b=$2 out=$3
	shift 3
	"$LLVM/bin/mlir-tblgen" "$b" "$@" -I "$ROOT/include" -I "$LLVM/include" \
		"$ROOT/include/ulm_lam/$td" -o "$GEN/$out"
}
tblgen UlmLamBase.td -gen-dialect-decls UlmLamDialect.h.inc -dialect=ulm_lam
tblgen UlmLamBase.td -gen-dialect-defs UlmLamDialect.cpp.inc -dialect=ulm_lam
tblgen UlmLamBase.td -gen-dialect-decls UlmLamkDialect.h.inc -dialect=ulm_lamk
tblgen UlmLamBase.td -gen-dialect-defs UlmLamkDialect.cpp.inc -dialect=ulm_lamk
tblgen UlmLamAttrs.td -gen-enum-decls UlmLamEnums.h.inc
tblgen UlmLamAttrs.td -gen-enum-defs UlmLamEnums.cpp.inc
tblgen UlmLamAttrs.td -gen-attrdef-decls UlmLamAttrs.h.inc -attrdefs-dialect=ulm_lam
tblgen UlmLamAttrs.td -gen-attrdef-defs UlmLamAttrs.cpp.inc -attrdefs-dialect=ulm_lam
tblgen UlmLamAttrs.td -gen-attrdef-decls UlmLamkAttrs.h.inc -attrdefs-dialect=ulm_lamk
tblgen UlmLamAttrs.td -gen-attrdef-defs UlmLamkAttrs.cpp.inc -attrdefs-dialect=ulm_lamk
tblgen UlmLamInterfaces.td -gen-op-interface-decls UlmLamInterfaces.h.inc
tblgen UlmLamInterfaces.td -gen-op-interface-defs UlmLamInterfaces.cpp.inc
tblgen UlmLamOps.td -gen-op-decls UlmLamOps.h.inc
tblgen UlmLamOps.td -gen-op-defs UlmLamOps.cpp.inc
tblgen UlmLamKernelOps.td -gen-op-decls UlmLamkOps.h.inc
tblgen UlmLamKernelOps.td -gen-op-defs UlmLamkOps.cpp.inc

"$LLVM/bin/clang++" -std=c++17 -O2 -fPIC -fno-rtti -shared \
	-I "$LLVM/include" -I "$ROOT/lib" -I "$GEN" "$ROOT"/lib/*.cpp \
	-L "$LLVM/lib" -lMLIR -lLLVM -Wl,-rpath,"$LLVM/lib" \
	-o "$OUT/UlmLamPasses.$EXT"
# compile_commands.json for clangd: C++ for lib/ (MLIR) and the C++ drivers
# (ulmBLAS), C11 for the C drivers
ULMBLAS_DIR="${ULMBLAS_DIR:-$HOME/git/ulmBLAS}"
{
	echo "["
	sep=""
	entry() { # file compiler args...
		local f=$1 c=$2 a=""
		shift 2
		for x in "$@"; do a="$a, \"$x\""; done
		printf '%s  {"directory": "%s", "file": "%s", "arguments": ["%s"%s, "-c", "%s"]}' \
			"$sep" "$ROOT" "$f" "$c" "$a" "$f"
		sep=$',\n'
	}
	for f in "$ROOT"/lib/*.cpp; do
		entry "$f" clang++ -std=c++17 -fno-rtti "-I$LLVM/include" "-I$ROOT/lib" "-I$GEN"
	done
	for f in "$ROOT"/examples/*/*.cc; do
		[ -e "$f" ] && entry "$f" clang++ -std=c++17 "-I$ULMBLAS_DIR"
	done
	for f in "$ROOT"/examples/*/*.c; do
		[ -e "$f" ] && entry "$f" cc -std=c11
	done
	printf '\n]\n'
} >"$ROOT/compile_commands.json"

echo "$OUT/UlmLamPasses.$EXT"
