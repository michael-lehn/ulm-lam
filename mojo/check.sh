#!/usr/bin/env bash
# make mojo: the Mojo backend of ulm.lam (ulm-lam --emit=mojo) on Apple
# Accelerate.
#   1. the kernels (ulm_lam/kernels.mojo) against plain loops
#   2. GLS (generalized least squares, examples/gls/02-normal.mlir):
#      generated Mojo against NumPy
#   3. the Python tests with the Mojo backend (ULM_LAM_BACKEND=mojo)
#   4. the example of the lectures (examples/sensor)
# Needs pixi (https://pixi.sh) for Mojo; NumPy for the system's python3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
if ! command -v pixi >/dev/null; then
	echo "make mojo: pixi not found (https://pixi.sh): skipped"
	exit 0
fi
cd "$HERE"
pixi install -q
mojo() { pixi run mojo "$@"; }
ACC=(-Xlinker -framework -Xlinker Accelerate)
mkdir -p build/gen build/gls

echo "== 1. kernels"
mojo build -I . "${ACC[@]}" tests/test_kernels.mojo -o build/test_kernels
build/test_kernels

echo "== 2. GLS, generated"
ULM="$("$ROOT/scripts/build-driver.sh")"
"$ULM" --backend=cblas --emit=mojo "$ROOT/examples/gls/02-normal.mlir" \
	-o build/gen/gls.mojo
mojo build -O3 -I . -I build/gen "${ACC[@]}" tests/gls_driver.mojo \
	-o build/gls_gen
build/gls_gen build/gls
python3 tests/check_gls.py build/gls

echo "== 3. Python tests, ULM_LAM_BACKEND=mojo"
(cd "$ROOT" && ULM_LAM_BACKEND=mojo python3 -m unittest discover \
	-s python/tests -q)

echo "== 4. the example of the lectures"
python3 examples/sensor/gen.py
mojo build -O3 -I . -I examples/sensor "${ACC[@]}" examples/sensor/main.mojo \
	-o build/sensor
build/sensor
