#!/usr/bin/env bash
# Reproduce every example of the companion; the outputs go to ../outputs.
# Run from anywhere:  doc/field-guide-companion/examples/run.sh
# The companion describes commit 4185542.  To run the examples against
# exactly that compiler, check it out separately and point ULM_LAM_ROOT at it:
#   git worktree add ../ulm-lam-4185542 4185542
#   ULM_LAM_ROOT=$PWD/../ulm-lam-4185542 doc/field-guide-companion/examples/run.sh
# Without ULM_LAM_ROOT the compiler of this checkout is used.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${ULM_LAM_ROOT:-$HERE/../../..}" && pwd)"
OUT="$HERE/../outputs"
mkdir -p "$OUT/gls"
make -C "$ROOT" tool >/dev/null
OPT="$ROOT/build/bin/ulm-lam-opt"
cd "$ROOT"
# the C++ on Julia's BLAS and the cache in build/julia, as at that commit
export ULM_LAM_BACKEND=lbt ULM_LAM_CACHE="$ROOT/build/julia"

# 1. GLS (generalized least squares) in Julia: traced IR (intermediate
#    representation), numbers, @code_lam, failures
julia --project=julia/UlmLAM "$HERE/gls.jl" >"$OUT/gls.out" 2>&1
# the stages the Julia call left in build/julia/gls, plus the two
# intermediate stages that ulm-lam-compile.sh does not keep
G=build/julia/gls
cp "$G/gls.mlir" "$OUT/gls/1-traced.mlir"
"$OPT" "$G/gls.mlir" --pass-pipeline='builtin.module(func.func(ulm-lam-normalize))' \
	-o "$OUT/gls/2-normal.mlir"
"$OPT" "$OUT/gls/2-normal.mlir" \
	--pass-pipeline='builtin.module(func.func(ulm-lam-infer-properties))' \
	-o "$OUT/gls/3-inferred.mlir"
"$OPT" "$OUT/gls/3-inferred.mlir" \
	--pass-pipeline='builtin.module(func.func(loop-invariant-code-motion, ulm-lam-select-kernels))' \
	-o "$OUT/gls/4-selected-with-props.mlir"
cp "$G/gls.selected.mlir" "$OUT/gls/5-selected.mlir"
cp "$G/gls.bufferized.mlir" "$OUT/gls/6-bufferized.mlir"
cp "$G/gls.cpp" "$OUT/gls/7-gls.cpp"

# 2. variants: dropped properties, assume, a loop, a shared product
julia --project=julia/UlmLAM "$HERE/variants.jl" >"$OUT/variants.out" 2>&1
cp build/julia/iterate/iterate.mlir "$OUT/iterate-traced.mlir"
"$OPT" build/julia/iterate/iterate.mlir \
	--pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, loop-invariant-code-motion))' \
	-o "$OUT/iterate-licm.mlir"
cp build/julia/iterate/iterate.selected.mlir "$OUT/iterate-selected.mlir"

# 3. a factorisation hoisted out of a loop that runs zero times
julia --project=julia/UlmLAM "$HERE/speculate.jl" >"$OUT/speculate.out" 2>&1

# 4. conditioning: the same formula, X almost rank deficient
julia --project=julia/UlmLAM "$HERE/conditioning.jl" >"$OUT/conditioning.out" 2>&1

# 5. verifiers: accepted and rejected IR
(cd "$HERE/verify" && for f in v[0-9]-*.mlir v10-*.mlir; do
	echo "### $f"
	"$OPT" "$f" -o /dev/null 2>&1 && echo "accepted" || true
done) >"$OUT/verify.out"
# dynamic sizes: kernel selection without and with ulm_lam.dims
(cd "$HERE/verify" && {
	echo "### v2-mul-dynamic.mlir through ulm-lam-select-kernels"
	"$OPT" v2-mul-dynamic.mlir --pass-pipeline='builtin.module(func.func(ulm-lam-select-kernels))' 2>&1 || true
	echo "### v11-dims-mismatch.mlir through the selection pipeline"
	"$OPT" v11-dims-mismatch.mlir --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' 2>&1 || true
}) >"$OUT/dims.out"
# effects: an unused factorisation on tensors and on memrefs, canonicalized
(cd "$HERE/verify" && "$OPT" v12-dead-potrf.mlir --canonicalize) >"$OUT/dead.out" 2>&1

# 6. the order of Q K^T V chosen at run time
"$OPT" test/select/runtime_order.mlir \
	--pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels), canonicalize)' \
	-o "$OUT/runtime_order.mlir"

# 7. versions
{
	echo "ulm-lam commit: $(git -C "$ROOT" rev-parse HEAD)"
	"$(brew --prefix llvm)/bin/mlir-opt" --version | head -2
	julia --version
	julia --project=julia/UlmLAM -e 'using LinearAlgebra; println("BLAS: ", BLAS.get_config())'
	"$(brew --prefix llvm)/bin/clang++" --version | head -1
	python3 --version
	sw_vers | head -2 | tr '\n' ' '; echo; uname -m
} >"$OUT/versions.out" 2>&1
echo "outputs in $OUT"
