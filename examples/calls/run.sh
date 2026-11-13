#!/usr/bin/env bash
# Functions calling functions (make calls): the examples in Julia and in
# Python; their IR after normalisation must be the same.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
julia --project=julia/UlmLAM examples/calls/calls.jl | tee build/calls_jl.out
python3 examples/calls/calls.py | tee build/calls_py.out
"$ROOT/scripts/build-tool.sh" >/dev/null
for m in calls calls_inout calls_lapack; do
	for f in jl py; do
		build/bin/ulm-lam-opt examples/calls/generated/${m}_$f.mlir \
			--pass-pipeline='builtin.module(func.func(ulm-lam-normalize))' \
			>build/${m}_$f.normalized.mlir
	done
	if diff -q build/${m}_jl.normalized.mlir build/${m}_py.normalized.mlir >/dev/null; then
		echo "PASS  $m: the same IR from Julia and from Python (normalised)"
	else
		echo "FAIL  $m: the IR from Julia and from Python differ"
		exit 1
	fi
	cp build/${m}_py.normalized.mlir examples/calls/generated/${m}_normalized.mlir
done
# the outputs of both must agree in their workspace sizes
grep "workspace" build/calls_jl.out >build/calls_ws_jl.txt
grep "workspace" build/calls_py.out >build/calls_ws_py.txt
diff build/calls_ws_jl.txt build/calls_ws_py.txt && echo "PASS  the same workspace sizes"
